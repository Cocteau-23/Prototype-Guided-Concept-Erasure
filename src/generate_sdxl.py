import argparse, os, csv, random
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPTokenizer
from contextlib import nullcontext 

from utils import ensure_dir, parse_dtype, read_prompts_from_csv, load_prototypes_payload, apply_erasure

def build_pipeline(args) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=parse_dtype(args.dtype), use_safetensors=True, variant="fp16" if parse_dtype(args.dtype) == torch.float16 else None
    )

    pipe.vae.to(dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if args.enable_xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except: pass

    if hasattr(pipe, "safety_checker"): pipe.safety_checker = None

    return pipe.to(args.device)

@torch.no_grad()
def encode_prompts_sdxl_custom(pipe: StableDiffusionXLPipeline, prompts):
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompts, prompt_2=None, device=pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=[""] * len(prompts) 
    )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--prototypes_path", type=str)
    ap.add_argument("--clip_model_path", type=str, default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_pos", type=float, default=7.5)
    ap.add_argument("--guidance_neg", type=float, default=20.0) 
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--prompt_batch_size", type=int, default=2)
    ap.add_argument("--mix_top_k", type=int, default=1)
    ap.add_argument("--mix_tau", type=float, default=0.0)
    ap.add_argument("--sim_threshold", type=float, default=0.15)
    ap.add_argument("--neg_disable_last_pct", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--enable_xformers", action="store_true", default=True)
    args = ap.parse_args()

    pipe = build_pipeline(args)
    device, dtype = pipe.device, pipe.unet.dtype
    prompts, csv_seeds = read_prompts_from_csv(args.csv)
    img_dir = ensure_dir(os.path.join(ensure_dir(args.out_dir), "images"))
    
    P, P_clip, _ = load_prototypes_payload(args.prototypes_path, device, dtype)
    
    (prompt_embeds_all, neg_embeds_all, pooled_pos_all, pooled_neg_all) = encode_prompts_sdxl_custom(pipe, prompts)
    
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path)
    clip_model_sim = CLIPModel.from_pretrained(args.clip_model_path).eval().to(device)
    
    with torch.no_grad():
        inputs = {k:v.to(device) for k,v in clip_tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").items()}
        prompt_clip_feats = F.normalize(clip_model_sim.get_text_features(**inputs), dim=-1).to(torch.float32)
        sims = prompt_clip_feats @ P_clip.T
        top1_all, sim_vals_all = sims.argmax(dim=1), sims.max(dim=1).values.tolist()
        P_ln_all = pipe.text_encoder_2.text_model.final_layer_norm(P)

    del clip_model_sim, inputs; torch.cuda.empty_cache()

    add_time_ids = torch.tensor([list((args.height, args.width) + (0,0) + (args.height, args.width))], dtype=dtype, device=device)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    
    total_images, rows, bs = 0, [], args.prompt_batch_size
    base_seed = args.seed if args.seed != 0 else random.randint(0, 10**9)
    do_autocast = (device.type == "cuda") and (dtype != torch.float32)

    for i0 in range(0, len(prompts), bs):
        i1 = min(len(prompts), i0 + bs)
        bsz = i1 - i0
        
        pe_batch, ne_batch = prompt_embeds_all[i0:i1], neg_embeds_all[i0:i1]
        
        topk_vals, topk_idx = torch.topk(sims[i0:i1], k=args.mix_top_k, dim=1)
        weights = torch.softmax(topk_vals / max(1e-6, args.mix_tau), dim=1).to(dtype)
        
        neg_proto_bigG = (weights.view(bsz, args.mix_top_k, 1, 1) * P_ln_all[topk_idx.reshape(-1)].reshape(bsz, args.mix_top_k, 77, 1280)).sum(dim=1) 
        neg_proto_combined = torch.cat([ne_batch[..., :768], neg_proto_bigG], dim=-1) 
        
        mask_use = topk_vals[:, 0] > args.sim_threshold
        if (~mask_use).any(): neg_proto_combined[~mask_use] = ne_batch[~mask_use]
            
        latents_list, seeds_list = [], []
        for bi in range(bsz):
            seed = csv_seeds[i0+bi] if csv_seeds else base_seed + total_images + bi
            seeds_list.append(seed)
            latents_list.append(torch.randn((1, 4, args.height//8, args.width//8), generator=torch.Generator(device).manual_seed(seed), device=device, dtype=dtype))
            
        latents = (torch.cat(latents_list, dim=0) * pipe.scheduler.init_noise_sigma).to(dtype)
        
        prompt_embeds_3 = torch.cat([ne_batch, pe_batch, neg_proto_combined], dim=0)
        added_cond_kwargs = {"text_embeds": torch.cat([pooled_neg_all[i0:i1], pooled_pos_all[i0:i1], pooled_neg_all[i0:i1]], dim=0), "time_ids": torch.cat([add_time_ids] * (bsz * 3), dim=0)}

        with torch.inference_mode():
            decay_until = int(args.num_inference_steps * (1.0 - args.neg_disable_last_pct))
            for step_idx, t in enumerate(pipe.scheduler.timesteps):
                latent_3 = torch.cat([pipe.scheduler.scale_model_input(latents, t)]*3, dim=0)
                
                with torch.autocast("cuda", dtype=dtype) if do_autocast else nullcontext():
                    noise_u, noise_pos, noise_proto = pipe.unet(latent_3, t, encoder_hidden_states=prompt_embeds_3, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0].chunk(3)
                
                noise_final = apply_erasure(
                    noise_u, noise_pos, noise_proto, 
                    args.guidance_pos, 
                    args.guidance_neg if step_idx <= decay_until else 0.0, 
                    erasure_threshold=0.01, rescale_phi=0.5
                )
                
                latents = pipe.scheduler.step(noise_final, t, latents).prev_sample
                
            imgs = (pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)

        for bi in range(bsz):
            img = (imgs[bi].cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            if args.height > 512:
                try: import cv2; img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                except: pass 
            Image.fromarray(img).save(os.path.join(img_dir, f"{total_images:06d}.png"))
            rows.append({"idx": total_images, "prompt": prompts[i0+bi], "seed": seeds_list[bi], "sim": float(sim_vals_all[i0+bi]), "proto_idx": int(topk_idx[bi, 0].item())})
            total_images += 1
            
    if rows:
        with open(os.path.join(args.out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=list(rows[0].keys())).writeheader()
            csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerows(rows)

if __name__ == "__main__":
    main()