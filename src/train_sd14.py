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
    ap.add_argument("--clip_model_path", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--pos_image_dir", type=str, help="Positive concept image dir")
    ap.add_argument("--neg_image_dir", type=str, help="Negative concept image dir")
    ap.add_argument("--output_path", type=str, help="Path to save the .pt file")

    ap.add_argument("--num_prototypes", type=int, default=16, help="Number of prototypes K")
    ap.add_argument("--train_steps", type=int, default=2000, help="Training steps")
    ap.add_argument("--lr", type=float, default=3e-2, help="Learning rate")
    ap.add_argument("--init_std", type=float, default=5e-2, help="Prototype initialization noise std")

    ap.add_argument("--shots", type=int, default=1, help="Samples per group")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    ap.add_argument("--soft_eot_tau", type=float, default=0.1, help="Soft EOT temperature")
    ap.add_argument("--lambda_shape", type=float, default=0.0, help="Shape regularization weight")

    ap.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision")
    args = ap.parse_args()

    set_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    clip_model      = CLIPModel.from_pretrained(args.clip_model_path).eval().to(device)
    tokenizer       = CLIPTokenizer.from_pretrained(args.clip_model_path)
    text_encoder    = CLIPTextModel.from_pretrained(args.clip_model_path).eval().to(device)
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path)

    eot_idx = get_eot_index_for_empty(tokenizer)
    print(f"[Info] EOT Index = {eot_idx}")

    assert args.pos_image_dir, "Must provide --pos_image_dir"
    has_neg = bool(args.neg_image_dir) and os.path.isdir(args.neg_image_dir)

    keys_pos, groups_pos, mode_pos = auto_build_groups(args.pos_image_dir, args.shots)
    if len(groups_pos) == 0:
        raise RuntimeError("No valid images detected in POS directory.")

    if has_neg:
        keys_neg, groups_neg, mode_neg = auto_build_groups(args.neg_image_dir, args.shots)
        print(f"[Grouping] POS: {len(groups_pos)} groups ({mode_pos}), NEG: {len(groups_neg)} groups ({mode_neg})")

        if mode_pos == "subdir" and mode_neg == "subdir":
            key_set = sorted(set(keys_pos) & set(keys_neg))
            if len(key_set) == 0:
                raise RuntimeError("No common subdirectories for pos/neg under subdir mode; check data.")
            mp = {k: i for i, k in enumerate(keys_pos)}
            mn = {k: i for i, k in enumerate(keys_neg)}
            groups_pos_aligned = [groups_pos[mp[k]] for k in key_set]
            groups_neg_aligned = [groups_neg[mn[k]] for k in key_set]
            print(f"[Grouping] Common key groups used: {len(key_set)}")
        else:
            gmin = min(len(groups_pos), len(groups_neg))
            if gmin == 0:
                raise RuntimeError("No valid groups detected, check directories.")
            groups_pos_aligned = groups_pos[:gmin]
            groups_neg_aligned = groups_neg[:gmin]
            print(f"[Grouping] Using first {gmin} groups aligned in order")
    else:
        mode_neg = None
        groups_pos_aligned = groups_pos
        groups_neg_aligned = None
        print(f"[Grouping] POS only: {len(groups_pos)} groups ({mode_pos}); clustering directly on POS")

    paths_pos_all = sum(groups_pos_aligned, [])
    Z_pos_all = embed_image_list(paths_pos_all, clip_model, image_processor,
                                 args.batch_size, device, args.amp, amp_dtype=torch.bfloat16)

    if has_neg:
        paths_neg_all = sum(groups_neg_aligned, [])
        Z_neg_all = embed_image_list(paths_neg_all, clip_model, image_processor,
                                     args.batch_size, device, args.amp, amp_dtype=torch.bfloat16)
    else:
        paths_neg_all = []
        Z_neg_all = None

    if has_neg:
        D = compute_delta_per_prompt_schemeD(groups_pos_aligned, groups_neg_aligned,
                                             Z_pos_all, paths_pos_all, Z_neg_all, paths_neg_all, device)
        assert D.shape[0] >= args.num_prototypes, f"Delta count {D.shape[0]} is less than K={args.num_prototypes}"
        targets = compute_kmeans_centers(D, args.num_prototypes, args.seed, name="KMeans on Delta").to(device)
        method_str = "Scheme D (mean(pos)-mean(neg)) + KMeans on Delta"
    else:
        assert Z_pos_all.shape[0] >= args.num_prototypes, f"POS samples {Z_pos_all.shape[0]} less than K={args.num_prototypes}"
        targets = compute_kmeans_centers(Z_pos_all, args.num_prototypes, args.seed, name="KMeans on POS").to(device)
        method_str = "POS-only KMeans centers"

    with torch.no_grad():
        inputs = tokenizer([""], padding="max_length", truncation=True,
                           max_length=tokenizer.model_max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        E_null_seq = text_encoder(**inputs).last_hidden_state.to(torch.float32)

    K = args.num_prototypes
    hidden_size = E_null_seq.shape[-1]
    with torch.no_grad():
        P0 = E_null_seq.expand(K, -1, -1).to(torch.float32)
    P = torch.nn.Parameter(P0 + args.init_std * torch.randn(K, 77, hidden_size, device=device, dtype=torch.float32))
    opt = torch.optim.AdamW([P], lr=args.lr)

    text_ln = text_encoder.text_model.final_layer_norm
    text_proj = clip_model.text_projection 

    print(f"[Train] Started training {K} prototypes for {args.train_steps} steps...")
    pbar = tqdm(range(args.train_steps), desc="Training (POS or Delta)")

    for step in pbar:
        if args.soft_eot_tau > 0:
            P_norm = text_ln(P)
            ref = P_norm[:, eot_idx:eot_idx+1, :]
            d2 = (P_norm - ref).pow(2).sum(-1)
            w = torch.softmax(-d2 / max(args.soft_eot_tau, 1e-6), dim=1)
            P_soft = (w.unsqueeze(-1) * P_norm).sum(1)
            if hasattr(text_proj, 'weight'):
                P_clip = text_proj(P_soft)
            else:
                W = text_proj if torch.is_tensor(text_proj) else text_proj.data
                P_clip = P_soft @ W.to(P_soft.device)
        else:
            P_norm = text_ln(P)
            P_eot = P_norm[:, eot_idx, :]
            if hasattr(text_proj, 'weight'):
                P_clip = text_proj(P_eot)
            else:
                W = text_proj if torch.is_tensor(text_proj) else text_proj.data
                P_clip = P_eot @ W.to(P_eot.device)
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
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "align": f"{loss_align.item():.4f}",
                "shape_w": f"{(args.lambda_shape * loss_shape).item():.4f}"
            })

    print(f"[Done] Final Loss: {loss.item():.6f}")

    with torch.no_grad():
        P_norm = text_ln(P)
        if args.soft_eot_tau > 0:
            ref = P_norm[:, eot_idx:eot_idx+1, :]
            d2  = (P_norm - ref).pow(2).sum(-1)
            w   = torch.softmax(-d2 / max(args.soft_eot_tau, 1e-6), dim=1)
            P_soft = (w.unsqueeze(-1) * P_norm).sum(1)
            if hasattr(text_proj, 'weight'):
                P_clip_final = text_proj(P_soft)
            else:
                W = text_proj if torch.is_tensor(text_proj) else text_proj.data
                P_clip_final = P_soft @ W.to(P_soft.device)
        else:
            P_eot = P_norm[:, eot_idx, :]
            if hasattr(text_proj, 'weight'):
                P_clip_final = text_proj(P_eot)
            else:
                W = text_proj if torch.is_tensor(text_proj) else text_proj.data
                P_clip_final = P_eot @ W.to(P_eot.device)
        P_clip_final = F.normalize(P_clip_final, dim=-1)

    meta = {
        "clip_model_path": args.clip_model_path,
        "eot_idx": int(eot_idx),
        "grouping_mode_pos": mode_pos,
        "grouping_mode_neg": mode_neg,
        "shots": int(args.shots),
        "num_groups_used": int(len(groups_pos_aligned)),
        "num_prototypes": int(K),
        "seed": int(args.seed),
        "method": method_str
    }
    payload = {
        "prototypes": P.detach().cpu().float(),
        "prototypes_clip": P_clip_final.detach().cpu(),
        "direction_centers": targets.detach().cpu(),
        "meta": meta,
    }
    out_dir = os.path.dirname(args.output_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"[Save] Prototypes saved to: {args.output_path}")

if __name__ == "__main__":

    main()