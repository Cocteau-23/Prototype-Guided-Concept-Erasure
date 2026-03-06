import argparse, os, time, random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPTokenizer, CLIPTextModel

from utils import (
    ensure_dir, parse_dtype, read_prompts_from_csv, 
    load_prototypes_payload, compute_prompt_clip_vectors_batch, 
    apply_erasure
)
BASE_MODEL = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL_PATH = "openai/clip-vit-large-patch14"

def build_pipeline(args) -> StableDiffusionPipeline:
    torch_dtype = parse_dtype(args.dtype)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    components = {}
    components["tokenizer"] = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)
    components["text_encoder"] = CLIPTextModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch_dtype, **components
    )
    pipe.vae.to(dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    if device == "cuda": torch.backends.cuda.matmul.allow_tf32 = True
    return pipe.to(device)

@torch.no_grad()
def encode_prompts_batch(pipe: StableDiffusionPipeline, prompts: List[str], text_ln: torch.nn.Module):
    device = pipe.device
    dtype = pipe.text_encoder.dtype
    
    text_inputs = pipe.tokenizer(prompts, padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    prompt_embeds_all = text_ln(pipe.text_encoder(text_inputs.input_ids.to(device))[0]).to(dtype)
    
    uncond_inputs = pipe.tokenizer([""], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    e_null_single = text_ln(pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]).to(dtype)
    return prompt_embeds_all, e_null_single

@torch.no_grad()
def compute_prototypes_clip_from_P(pipe, P, meta):
    device = P.device
    text_ln = pipe.text_encoder.text_model.final_layer_norm
    eot_idx = int(meta.get("eot_idx", P.shape[1]-1))
    P_eot = text_ln(P)[:, eot_idx, :]
    
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).eval().to(device)
    W = clip_model.text_projection if torch.is_tensor(clip_model.text_projection) else clip_model.text_projection.data
    
    P_clip = P_eot @ W.to(device) if not hasattr(clip_model.text_projection, "weight") else clip_model.text_projection(P_eot.to(torch.float32))
    return F.normalize(P_clip, dim=-1).to(torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototypes_path", type=str)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--guidance_pos", type=float, default=7.5)
    ap.add_argument("--guidance_neg", type=float, default=20.0)
    ap.add_argument("--num_images_per_prompt", type=int, default=1)
    ap.add_argument("--prompt_batch_size", type=int, default=4)
    ap.add_argument("--sim_threshold", type=float, default=0.15)
    ap.add_argument("--neg_disable_last_pct", type=float, default=0.4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pipe = build_pipeline(args)
    device = pipe.device
    dtype_text = pipe.text_encoder.dtype
    text_ln = pipe.text_encoder.text_model.final_layer_norm

    prompts, csv_seeds = read_prompts_from_csv(args.csv)
    img_dir = ensure_dir(os.path.join(ensure_dir(args.out_dir), "images"))

    P, P_clip_saved, meta = load_prototypes_payload(args.prototypes_path, device, dtype_text)
    P_clip = P_clip_saved if P_clip_saved is not None else compute_prototypes_clip_from_P(pipe, P, meta)
        
    with torch.inference_mode():
        prompt_embeds_all, e_null_single = encode_prompts_batch(pipe, prompts, text_ln)
        clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).eval().to(device)
        
        prompt_clip_all = compute_prompt_clip_vectors_batch(clip_tokenizer, clip_model, prompts, device)
        sims = prompt_clip_all @ P_clip.T
        top1_all, sim_vals_all = sims.argmax(dim=1), sims.max(dim=1).values.tolist()
        P_ln_all = text_ln(P)

    base_seed = random.randint(1, 10**9) if (csv_seeds is None and args.seed == 0) else args.seed
    
    pipe.scheduler.set_timesteps(30, device=device)
    
    total_images, rows = 0, []
    bs, num_per = max(1, args.prompt_batch_size), max(1, args.num_images_per_prompt)

    for i0 in range(0, len(prompts), bs):
        i1 = min(len(prompts), i0 + bs)
        bsz = i1 - i0

        pe_batch, e_null_batch = prompt_embeds_all[i0:i1], e_null_single.expand(bsz, -1, -1)
        
        top1_vals, top1_idx = torch.max(sims[i0:i1], dim=1)
        neg_batch = P_ln_all[top1_idx].to(dtype_text)
        
        mask_use_proto = top1_vals > args.sim_threshold
        if (~mask_use_proto).any():
            neg_batch[~mask_use_proto] = e_null_batch[~mask_use_proto].to(neg_batch.dtype)

        B = bsz * num_per
        pe_rep, en_rep, neg_rep = pe_batch.repeat_interleave(num_per, 0), e_null_batch.repeat_interleave(num_per, 0), neg_batch.repeat_interleave(num_per, 0)

        latents_list, seeds_list = [], []
        for bi in range(B):
            pidx = i0 + (bi // num_per)
            current_seed = csv_seeds[pidx] + (bi % num_per) if csv_seeds else base_seed + total_images + bi
            seeds_list.append(current_seed)
            gen = torch.Generator(device=device).manual_seed(current_seed)
            
            latents_list.append(torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=gen, device=device, dtype=pipe.unet.dtype))
            
        latents = (torch.cat(latents_list, dim=0) * pipe.scheduler.init_noise_sigma).to(pipe.unet.dtype)
        generators_list = [torch.Generator(device=device).manual_seed(s) for s in seeds_list]

        with torch.inference_mode():
            decay_until = max(1, int(len(pipe.scheduler.timesteps) * (1.0 - args.neg_disable_last_pct)))
            for step_idx, t in enumerate(pipe.scheduler.timesteps):
                latent_in = pipe.scheduler.scale_model_input(latents, t)
                noise_3 = pipe.unet(torch.cat([latent_in]*3, dim=0), t, encoder_hidden_states=torch.cat([en_rep, pe_rep, neg_rep], dim=0).to(dtype_text)).sample
                noise_u, noise_pos, noise_proto = noise_3.chunk(3)
                
                noise = apply_erasure(
                    noise_u, noise_pos, noise_proto, 
                    args.guidance_pos, 
                    args.guidance_neg if step_idx <= decay_until else 0.0, 
                    erasure_threshold=0.005, rescale_phi=0.5
                )
                
                latents = pipe.scheduler.step(noise, t, latents, generator=generators_list).prev_sample.to(pipe.unet.dtype)

            imgs = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(torch.float32)).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1).to(torch.float32)

        for bi in range(B):
            img = imgs[bi].detach().cpu().permute(1, 2, 0).numpy()
            if np.isnan(img).any() or np.isinf(img).any(): img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            
            path = os.path.join(img_dir, f"{total_images:06d}.png")
            Image.fromarray((img * 255).round().astype("uint8")).save(path)

            pidx = i0 + (bi // num_per)
            rows.append({
                "idx": total_images, "prompt": prompts[pidx], "seed": seeds_list[bi],
                "proto_index": int(top1_all[pidx]), "proto_sim": float(sim_vals_all[pidx])
            })
            total_images += 1

    import csv
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=list(rows[0].keys())).writeheader()
        csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerows(rows)
    print(f"Done. Output: {args.out_dir}")

if __name__ == "__main__":
    main()