import argparse, os, csv, torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from transformers import CLIPModel, CLIPTokenizer

from utils import ensure_dir, read_prompts_from_csv, load_prototypes_payload, apply_erasure

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="/root/autodl-tmp/models/sd3.5-large")
    ap.add_argument("--prototypes_path", type=str)
    ap.add_argument("--clip_model_path", type=str, default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    ap.add_argument("--csv", type=str)
    ap.add_argument("--out_dir", type=str)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=4.5)
    ap.add_argument("--guidance_neg", type=float, default=20.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim_threshold", type=float, default=0.15)
    args = ap.parse_args()

    dtype = torch.float16 
    pipe = StableDiffusion3Pipeline.from_pretrained(args.base_model, torch_dtype=dtype, device_map="balanced")
    main_device = pipe.device

    sim_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    sim_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path)
    sim_model = CLIPModel.from_pretrained(args.clip_model_path).eval().to(sim_device)

    _, P_vectors, _ = load_prototypes_payload(args.prototypes_path, sim_device, dtype) 
    prompts, seeds = read_prompts_from_csv(args.csv)
    img_dir = ensure_dir(os.path.join(args.out_dir, "images"))
    
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    print("Pre-calculating similarities...")
    all_sims = []
    with torch.no_grad():
        for i in range(0, len(prompts), 8):
            batch_p = prompts[i:i+8]
            inputs = sim_tokenizer(batch_p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(sim_device)
            feats = sim_model.get_text_features(**inputs)
            sim = (feats / feats.norm(dim=-1, keepdim=True)).to(dtype) @ P_vectors.T
            all_sims.extend(sim.cpu())

    del sim_model, sim_tokenizer
    torch.cuda.empty_cache()

    rows, total_idx = [], 0
    for i in range(0, len(prompts), args.batch_size):
        bs = min(args.batch_size, len(prompts) - i)
        batch_prompts = prompts[i:i+bs]
        batch_seeds = seeds[i:i+bs] if seeds else [args.seed + total_idx + j for j in range(bs)]
        
        (pos_prompt_embeds, pos_neg_embeds, pos_pooled, pos_neg_pooled) = pipe.encode_prompt(
            prompt=batch_prompts, prompt_2=batch_prompts, prompt_3=batch_prompts,
            device=main_device, do_classifier_free_guidance=True
        )
        
        current_sims = torch.stack(all_sims[i:i+bs]).to(main_device, dtype=dtype)
        vals, idxs = torch.topk(current_sims, k=1, dim=1)
        selected_protos = P_vectors.to(main_device)[idxs.squeeze(1)]
        
        proto_pooled_input = pos_neg_pooled.clone()
        mask = (vals.squeeze(1) > args.sim_threshold)
        
        if mask.any():
            if proto_pooled_input.shape[-1] == 2048: proto_pooled_input[mask, 768:] = selected_protos[mask]
            else: proto_pooled_input[mask] = selected_protos[mask]
        
        latents = torch.cat([pipe.prepare_latents(1, 16, args.height, args.width, dtype, main_device, generator=torch.Generator(device=main_device).manual_seed(s)) for s in batch_seeds], dim=0)

        pipe.scheduler.set_timesteps(args.steps, device=main_device)
        prompt_embeds_3 = torch.cat([pos_neg_embeds, pos_prompt_embeds, pos_neg_embeds], dim=0)
        pooled_embeds_3 = torch.cat([pos_neg_pooled, pos_pooled, proto_pooled_input], dim=0)
        
        with torch.inference_mode():
            for t in pipe.scheduler.timesteps:
                with torch.autocast("cuda", dtype=dtype):
                    noise_pred = pipe.transformer(
                        hidden_states=torch.cat([latents] * 3), timestep=torch.tensor([t] * (bs * 3), device=main_device),
                        encoder_hidden_states=prompt_embeds_3, pooled_projections=pooled_embeds_3, return_dict=False
                    )[0].to(main_device)

                noise_u, noise_pos, noise_proto = noise_pred.chunk(3)
                
                noise_final = apply_erasure(
                    noise_u, noise_pos, noise_proto, 
                    args.guidance_scale, args.guidance_neg, 
                    erasure_threshold=0.01, rescale_phi=0.0
                )
                
                latents = pipe.scheduler.step(noise_final, t, latents, return_dict=False)[0]

        with torch.no_grad():
            latents = ((latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor).to(dtype=pipe.vae.dtype, device=main_device)
            image = pipe.image_processor.postprocess(pipe.vae.decode(latents, return_dict=False)[0], output_type="pil")

        for j, img in enumerate(image):
            img.save(os.path.join(img_dir, f"{total_idx:06d}.png"))
            rows.append({"idx": total_idx, "prompt": batch_prompts[j], "seed": batch_seeds[j], "proto_sim": float(vals[j].item()), "applied": bool(mask[j].item())})
            total_idx += 1
            
    if rows:
        with open(os.path.join(args.out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=list(rows[0].keys())).writeheader()
            csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerows(rows)

if __name__ == "__main__":
    main()