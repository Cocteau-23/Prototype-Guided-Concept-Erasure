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
    
    ap.add_argument("--soft_eot_tau", type=float, default=0.1)
    ap.add_argument("--lambda_shape", type=float, default=0.0)
    
    ap.add_argument("--amp", action="store_true", default=True)

    args = ap.parse_args()

    set_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[Info] Loading CLIP model from: {args.clip_model_path}")
    print("[Info] For SDXL, ensure this is the OpenCLIP ViT-bigG/14 model.")

    # 1) CLIP (Load BigG for SDXL)
    try:
        clip_model = CLIPModel.from_pretrained(args.clip_model_path).eval().to(device)
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path)
        text_encoder = CLIPTextModel.from_pretrained(args.clip_model_path).eval().to(device)
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path)
    except Exception as e:
        print(f"[Error] Failed to load CLIP model: {e}")
        print("Tip: Ensure you have access to 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'")
        return

    # 2) EOT
    eot_idx = get_eot_index_for_empty(tokenizer)
    print(f"[Info] EOT Index = {eot_idx}")

    # 3) Grouping
    assert args.pos_image_dir, "Must provide --pos_image_dir"
    has_neg = bool(args.neg_image_dir) and os.path.isdir(args.neg_image_dir)

    keys_pos, groups_pos, mode_pos = auto_build_groups(args.pos_image_dir, args.shots)
    if len(groups_pos) == 0:
        raise RuntimeError("No valid images detected in POS directory.")

    if has_neg:
        keys_neg, groups_neg, mode_neg = auto_build_groups(args.neg_image_dir, args.shots)
        print(f"[Grouping] POS: {len(groups_pos)} groups, NEG: {len(groups_neg)} groups")

        if mode_pos == "subdir" and mode_neg == "subdir":
            key_set = sorted(set(keys_pos) & set(keys_neg))
            if len(key_set) == 0:
                raise RuntimeError("No common subdirectories for pos/neg under subdir mode; check data.")
            mp = {k: i for i, k in enumerate(keys_pos)}
            mn = {k: i for i, k in enumerate(keys_neg)}
            groups_pos_aligned = [groups_pos[mp[k]] for k in key_set]
            groups_neg_aligned = [groups_neg[mn[k]] for k in key_set]
        else:
            gmin = min(len(groups_pos), len(groups_neg))
            if gmin == 0: raise RuntimeError("No valid groups detected.")
            groups_pos_aligned = groups_pos[:gmin]
            groups_neg_aligned = groups_neg[:gmin]
    else:
        mode_neg = None
        groups_pos_aligned = groups_pos
        groups_neg_aligned = None

    # 4) Precompute Embeddings
    paths_pos_all = sum(groups_pos_aligned, [])
    Z_pos_all = embed_image_list(paths_pos_all, clip_model, image_processor,
                                 args.batch_size, device, args.amp, amp_dtype=torch.float16)

    if has_neg:
        paths_neg_all = sum(groups_neg_aligned, [])
        Z_neg_all = embed_image_list(paths_neg_all, clip_model, image_processor,
                                     args.batch_size, device, args.amp, amp_dtype=torch.float16)
    else:
        paths_neg_all = []
        Z_neg_all = None

    # 5) Targets
    if has_neg:
        D = compute_delta_per_prompt_schemeD(groups_pos_aligned, groups_neg_aligned,
                                             Z_pos_all, paths_pos_all, Z_neg_all, paths_neg_all, device)
        targets = compute_kmeans_centers(D, args.num_prototypes, args.seed, name="KMeans on Delta").to(device)
        method_str = "Scheme D (mean(pos)-mean(neg)) + KMeans on Delta"
    else:
        targets = compute_kmeans_centers(Z_pos_all, args.num_prototypes, args.seed, name="KMeans on POS").to(device)
        method_str = "POS-only KMeans centers"

    # 6) Initialize Prototypes
    with torch.no_grad():
        inputs = tokenizer([""], padding="max_length", truncation=True,
                           max_length=tokenizer.model_max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        E_null_seq = text_encoder(**inputs).last_hidden_state.to(torch.float32)

    K = args.num_prototypes
    hidden_size = E_null_seq.shape[-1] # SDXL BigG should be 1280
    print(f"[Info] Prototype hidden dimension: {hidden_size}")
    
    with torch.no_grad():
        P0 = E_null_seq.expand(K, -1, -1).to(torch.float32)
    P = torch.nn.Parameter(P0 + args.init_std * torch.randn(K, 77, hidden_size, device=device, dtype=torch.float32))
    opt = torch.optim.AdamW([P], lr=args.lr)

    text_ln = text_encoder.text_model.final_layer_norm
    text_proj = clip_model.text_projection

    # 7) Train
    pbar = tqdm(range(args.train_steps), desc="Training Prototypes")

    for step in pbar:
        P_norm = text_ln(P)
        if args.soft_eot_tau > 0:
            ref = P_norm[:, eot_idx:eot_idx+1, :]
            d2 = (P_norm - ref).pow(2).sum(-1)
            w = torch.softmax(-d2 / max(args.soft_eot_tau, 1e-6), dim=1)
            P_soft = (w.unsqueeze(-1) * P_norm).sum(1)
            to_proj = P_soft
        else:
            to_proj = P_norm[:, eot_idx, :]
            
        if hasattr(text_proj, 'weight'):
            P_clip = text_proj(to_proj)
        else:
            W = text_proj if torch.is_tensor(text_proj) else text_proj.data
            P_clip = to_proj @ W.to(to_proj.device)
        P_clip = F.normalize(P_clip, dim=-1)

        loss_align = (1.0 - F.cosine_similarity(P_clip, targets, dim=1)).mean()

        loss_shape = torch.tensor(0.0, device=device)
        if args.lambda_shape > 0:
            P_norm_tokens = text_ln(P)
            mu_t  = E_null_seq.mean(dim=1, keepdim=True)
            std_t = E_null_seq.std(dim=1, keepdim=True) + 1e-6
            mu_p  = P_norm_tokens.mean(dim=1, keepdim=True)
            std_p = P_norm_tokens.std(dim=1, keepdim=True) + 1e-6
            loss_shape = F.mse_loss(mu_p, mu_t.expand_as(mu_p)) + F.mse_loss(std_p, std_t.expand_as(std_p))

        loss = loss_align + args.lambda_shape * loss_shape

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([P], max_norm=1.0)
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "align": f"{loss_align.item():.4f}"})

    # 8) Save
    with torch.no_grad():
        P_norm = text_ln(P)
        if args.soft_eot_tau > 0:
            ref = P_norm[:, eot_idx:eot_idx+1, :]
            d2 = (P_norm - ref).pow(2).sum(-1)
            w = torch.softmax(-d2 / max(args.soft_eot_tau, 1e-6), dim=1)
            to_proj = (w.unsqueeze(-1) * P_norm).sum(1)
        else:
            to_proj = P_norm[:, eot_idx, :]
            
        if hasattr(text_proj, 'weight'):
            P_clip_final = text_proj(to_proj)
        else:
            W = text_proj if torch.is_tensor(text_proj) else text_proj.data
            P_clip_final = to_proj @ W.to(to_proj.device)
        P_clip_final = F.normalize(P_clip_final, dim=-1)

    meta = {
        "clip_model_path": args.clip_model_path,
        "eot_idx": int(eot_idx),
        "num_prototypes": int(K),
        "hidden_size": int(hidden_size),
        "method": method_str
    }
    payload = {
        "prototypes": P.detach().cpu().float(),
        "prototypes_clip": P_clip_final.detach().cpu(),
        "meta": meta,
    }
    out_dir = os.path.dirname(args.output_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"[Save] SDXL Prototypes saved to: {args.output_path}")

if __name__ == "__main__":
    main()