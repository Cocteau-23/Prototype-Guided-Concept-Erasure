import argparse
import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate images from paired prompts using SD 1.4")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the generated prompts.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/test", help="Base directory to save generated images")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate per prompt")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale (CFG)")
    args = parser.parse_args()

    out_dir_on = os.path.join(args.out_dir, "on")
    out_dir_off = os.path.join(args.out_dir, "off")
    os.makedirs(out_dir_on, exist_ok=True)
    os.makedirs(out_dir_off, exist_ok=True)

    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    if "pair_id" not in df.columns or "variant" not in df.columns:
        raise ValueError("CSV must contain 'pair_id' and 'variant' columns.")
    
    grouped = df.groupby("pair_id")

    print("Loading Stable Diffusion 1.4 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)

    print(f"Starting generation... (Generating {args.num_images} images per prompt)")
    
    for pair_id, group in tqdm(grouped, desc="Processing Pairs"):
        if len(group) != 2:
            print(f"Warning: pair_id {pair_id} does not have exactly 2 rows. Skipping.")
            continue
            
        on_row = group[group["variant"] == "on"].iloc[0]
        off_row = group[group["variant"] == "off"].iloc[0]
        
        prompt_on = on_row["prompt"]
        prompt_off = off_row["prompt"]
        
        base_seed = int(on_row["seed"])
        category = on_row["category"]
        
        for i in range(args.num_images):
            current_seed = base_seed + i
            safe_pair_id = str(pair_id).replace('#', '_')
            
            generator_on = torch.Generator(device=device).manual_seed(current_seed)
            img_on = pipe(
                prompt=prompt_on, 
                generator=generator_on, 
                num_inference_steps=args.steps, 
                guidance_scale=args.cfg
            ).images[0]
            
            filename_on = f"{category}_{safe_pair_id}_img{i}_seed{current_seed}.png"
            img_on.save(os.path.join(out_dir_on, filename_on))
            
            generator_off = torch.Generator(device=device).manual_seed(current_seed)
            img_off = pipe(
                prompt=prompt_off, 
                generator=generator_off, 
                num_inference_steps=args.steps, 
                guidance_scale=args.cfg
            ).images[0]
            
            filename_off = f"{category}_{safe_pair_id}_img{i}_seed{current_seed}.png"
            img_off.save(os.path.join(out_dir_off, filename_off))

    print(f"\n[Done] All images saved to:")
    print(f"  - ON images:  {out_dir_on}")
    print(f"  - OFF images: {out_dir_off}")

if __name__ == "__main__":
    main()