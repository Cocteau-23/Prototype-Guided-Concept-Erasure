import argparse
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from utils import (
    set_seed_all, get_eot_index_for_empty, auto_build_groups, 
    embed_image_list, compute_delta_per_prompt_schemeD, compute_kmeans_centers
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_model_path", type=str, default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    ap.add_argument("--pos_image_dir", type=str, help="Positive image dir")
    ap.add_argument("--neg_image_dir", type=str, help="Negative image dir")
    ap.add_argument("--output_path", type=str)

    ap.add_argument("--num_prototypes", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--init_std", type=float, default=5e-2)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--soft_eot_tau", type=float, default=0.1)

    args = ap.parse_args()

    set_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Info] Loading CLIP (BigG) from: {args.clip_model_path}")
    try:
        clip_model = CLIPModel.from_pretrained(args.clip_model_path).eval().to(device)
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path)
        text_encoder = CLIPTextModel.from_pretrained(args.clip_model_path).eval().to(device)
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path)
    except Exception as e:
        print(f"[Error] {e}")
        return

    keys_pos, groups_pos, _ = auto_build_groups(args.pos_image_dir, args.shots)
    if not groups_pos: raise RuntimeError("No POS images")
    
    paths_pos = sum(groups_pos, [])
    Z_pos = embed_image_list(paths_pos, clip_model, image_processor, args.batch_size, device, args.amp, amp_dtype=torch.float16)
    
    Z_neg = None
    paths_neg = []
    groups_neg_aligned = []
    
    if args.neg_image_dir:
        keys_neg, groups_neg, _ = auto_build_groups(args.neg_image_dir, args.shots)
        min_len = min(len(groups_pos), len(groups_neg))
        groups_pos = groups_pos[:min_len]
        groups_neg_aligned = groups_neg[:min_len]
        paths_neg = sum(groups_neg_aligned, [])
        Z_neg = embed_image_list(paths_neg, clip_model, image_processor, args.batch_size, device, args.amp, amp_dtype=torch.float16)

    if Z_neg is not None:
        D = compute_delta_per_prompt_schemeD(groups_pos, groups_neg_aligned, Z_pos, paths_pos, Z_neg, paths_neg, device)
        targets = compute_kmeans_centers(D, args.num_prototypes, args.seed).to(device)
    else:
        targets = compute_kmeans_centers(Z_pos, args.num_prototypes, args.seed).to(device)

    eot_idx = get_eot_index_for_empty(tokenizer)
    with torch.no_grad():
        null_ids = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")["input_ids"].to(device)
        E_null = text_encoder(null_ids).last_hidden_state
    
    K = args.num_prototypes
    dim = E_null.shape[-1] # 1280
    
    P = torch.nn.Parameter(E_null.expand(K, -1, -1) + args.init_std * torch.randn(K, 77, dim, device=device))
    opt = torch.optim.AdamW([P], lr=args.lr)
    
    text_proj = clip_model.text_projection
    text_ln = text_encoder.text_model.final_layer_norm

    print(f"Training {K} prototypes (dim={dim}) for {args.train_steps} steps...")
    pbar = tqdm(range(args.train_steps))
    
    for step in pbar:
        P_norm = text_ln(P)
        feat = P_norm[:, eot_idx, :]
            
        if hasattr(text_proj, 'weight'):
            feat_proj = text_proj(feat)
        else:
            feat_proj = feat @ text_proj.data.to(device)
            
        feat_proj = F.normalize(feat_proj, dim=-1)
        
        # Loss
        loss = (1.0 - F.cosine_similarity(feat_proj, targets, dim=1)).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    meta = {
        "clip_model_path": args.clip_model_path,
        "num_prototypes": K,
        "dim": dim,
        "desc": "SD3.5 BigG Prototypes"
    }
    payload = {
        "prototypes": P.detach().cpu(), # [K, 77, 1280]
        "prototypes_clip": feat_proj.detach().cpu(), # [K, 1280]
        "meta": meta
    }
    torch.save(payload, args.output_path)
    print(f"Done. Saved to {args.output_path}")

if __name__ == "__main__":
    main()