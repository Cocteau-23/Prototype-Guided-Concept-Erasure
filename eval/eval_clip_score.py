import argparse
import os
import csv
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

@torch.no_grad()
def get_clip_score(model, processor, image: Image.Image, prompt: str, device: str):
    inputs = processor(
        text=[prompt], 
        images=[image], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    
    img_feat = outputs.image_embeds
    txt_feat = outputs.text_embeds
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    
    return torch.sum(img_feat * txt_feat, dim=-1).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    args = ap.parse_args()

    if not os.path.isfile(args.in_csv):
        raise FileNotFoundError(f"missing: {args.in_csv}")
    
    out_csv = args.out_csv or (os.path.splitext(args.in_csv)[0] + "_clip.csv")
    base_dir = args.images_root if args.images_root else os.path.dirname(os.path.abspath(args.in_csv))

    model = CLIPModel.from_pretrained(args.model_id).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_id)

    scores_sum = 0.0
    scores_count = 0

    with open(args.in_csv, "r", encoding="utf-8") as rf, \
         open(out_csv, "w", encoding="utf-8", newline="") as wf:
        
        reader = csv.DictReader(rf)
        fieldnames = list(reader.fieldnames)
        if "clip_score" not in fieldnames: 
            fieldnames.append("clip_score")
            
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            prompt = row.get("prompt", "")
            p = row.get("image_path")
            
            if not p:
                writer.writerow(row)
                continue
                
            full_path = p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
            
            try:
                img = Image.open(full_path).convert("RGB")
                score = get_clip_score(model, processor, img, prompt, args.device)
                row["clip_score"] = float(score)
                scores_sum += score
                scores_count += 1
            except Exception as e:
                row["clip_score"] = None
                
            writer.writerow(row)

    mean_score = (scores_sum / scores_count) if scores_count > 0 else float('nan')
    print(f"mean score: {mean_score:.6f}")

if __name__ == "__main__":
    main()