import argparse, os, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import csv
from scipy.linalg import sqrtm

def list_images_from_dir(d):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = []
    for fn in os.listdir(d):
        if fn.lower().endswith(exts):
            files.append(os.path.join(d, fn))
    return sorted(files)

def list_images_from_csv(csv_path, images_root=None):
    files = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("image_path")
            if p:
                files.append(os.path.join(images_root, p) if images_root else p)
    return files

class InceptionPool(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        m.eval().to(device)
        self.m = m
        self.device = device
        self.tx = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def encode(self, paths):
        feats = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                x = self.tx(img).unsqueeze(0).to(self.device)
                f = self.m(x)
                f = f.squeeze(0).cpu().numpy()
                feats.append(f)
            except Exception as e:
                pass
        return np.stack(feats, axis=0)

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)

def main():
    ap = argparse.ArgumentParser()
    
    real_group = ap.add_mutually_exclusive_group(required=True)
    real_group.add_argument("--real_dir", type=str)
    real_group.add_argument("--real_stats", type=str)
    
    gen_group = ap.add_mutually_exclusive_group(required=True)
    gen_group.add_argument("--gen_dir", type=str)
    gen_group.add_argument("--gen_csv", type=str)
    
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()
        
    pool = InceptionPool(device=args.device)

    if args.real_stats:
        try:
            data = np.load(args.real_stats)
            mu1 = data['mu']
            sigma1 = data['sigma']
            real_count = int(data.get('count', 0))
        except Exception as e:
            return
    else:
        real_paths = list_images_from_dir(args.real_dir)
        f_real = pool.encode(real_paths)
        mu1, sigma1 = f_real.mean(axis=0), np.cov(f_real, rowvar=False)
        real_count = int(len(f_real))

    if args.gen_dir:
        gen_paths = list_images_from_dir(args.gen_dir)
    else:
        gen_paths = list_images_from_csv(args.gen_csv, images_root=args.images_root)

    f_gen = pool.encode(gen_paths)
    mu2, sigma2 = f_gen.mean(axis=0), np.cov(f_gen, rowvar=False)
    gen_count = int(len(f_gen))
    
    print("\nCalculating FID...")
    fid = float(frechet_distance(mu1, sigma1, mu2, sigma2))
    print(f"FID: {fid:.4f} (real={real_count}, gen={gen_count})")

    if args.out_json:
        if os.path.dirname(args.out_json):
             os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
             
        with open(args.out_json, "w", encoding="utf-8") as jf:
            json.dump({
                "fid": fid,
                "real_count": real_count,
                "gen_count": gen_count
            }, jf, indent=2, ensure_ascii=False)
        print(f"[Saved] {args.out_json}")


if __name__ == "__main__":
    main()