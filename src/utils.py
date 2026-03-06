import os
import random
import csv
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor
from sklearn.cluster import MiniBatchKMeans

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def parse_dtype(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return torch.float32

def read_prompts_from_csv(csv_path: str) -> Tuple[List[str], Optional[List[int]]]:
    prompts: List[str] = []
    seeds: List[int] = []
    has_seed_column = False
    
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [f.strip() for f in reader.fieldnames] if reader.fieldnames else []
        
        prompt_key = "target_prompt" if "target_prompt" in fieldnames else "prompt"
            
        if "seed" in fieldnames:
            has_seed_column = True

        for row in reader:
            p = row.get(prompt_key, "").strip()
            if p:
                prompts.append(p)
                if has_seed_column:
                    try:
                        seeds.append(int(row["seed"]))
                    except (ValueError, TypeError):
                        pass
    
    seed_list = seeds if (has_seed_column and len(seeds) == len(prompts)) else None
    return prompts, seed_list

def load_prototypes_payload(path: str, device: torch.device, dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu")
    
    P = obj.get("prototypes", None)
    if P is not None:
        P = P.to(device=device, dtype=dtype)
    
    P_clip = obj.get("prototypes_clip", None)
    P_clip = P_clip.to(device=device, dtype=torch.float32)
    P_clip = F.normalize(P_clip, dim=-1)

        
    meta = obj.get("meta", {})
    return P, P_clip, meta

@torch.no_grad()
def compute_prompt_clip_vectors_batch(tokenizer, clip_model, prompts: List[str], device: torch.device) -> torch.Tensor:
    inputs = tokenizer(
        prompts, padding="max_length", truncation=True,
        max_length=77, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_feats = clip_model.get_text_features(**inputs)
    text_feats = F.normalize(text_feats, dim=-1).to(torch.float32)
    return text_feats

def apply_erasure(
    noise_u: torch.Tensor, 
    noise_pos: torch.Tensor, 
    noise_proto: torch.Tensor, 
    guidance_pos: float, 
    guidance_neg: float, 
    erasure_threshold: float = 0.01, 
    rescale_phi: float = 0.5
) -> torch.Tensor:
    
    d_pos = noise_pos - noise_u
    d_neg = noise_proto - noise_u 
    
    d_neg_abs = d_neg.abs()
    mask = (d_neg_abs > erasure_threshold).float()
    d_neg_masked = d_neg * mask
    
    noise_erased = noise_u + guidance_pos * d_pos - guidance_neg * d_neg_masked
    
    if rescale_phi > 0.0:
        noise_plain = noise_u + guidance_pos * d_pos
        std_plain = noise_plain.std([1, 2, 3], keepdim=True)
        std_erased = noise_erased.std([1, 2, 3], keepdim=True)
        
        target_std = std_plain / (std_erased + 1e-8)
        noise_rescaled = noise_erased * target_std
        
        noise_final = rescale_phi * noise_rescaled + (1.0 - rescale_phi) * noise_erased
        return noise_final
        
    return noise_erased





def set_seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

def get_eot_index_for_empty(tokenizer: CLIPTokenizer) -> int:
    ids = tokenizer([""], padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt")["input_ids"][0].tolist()
    eot_id = tokenizer.eos_token_id
    eot_positions = [i for i, t in enumerate(ids) if t == eot_id]
    return eot_positions[-1] if len(eot_positions) > 0 else len(ids) - 1

def list_sorted_images(image_dir: str) -> List[str]:
    exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(exts)])
    return [os.path.join(image_dir, f) for f in files]

def find_prompt_subdirs(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir): 
        return []
    cand = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))])
    subdirs = []
    for d in cand:
        imgs = list_sorted_images(d)
        if len(imgs) > 0:
            subdirs.append(d)
    return subdirs

class ImageListDataset(Dataset):
    def __init__(self, image_paths: List[str], processor: CLIPImageProcessor):
        self.paths = image_paths
        self.processor = processor
        
    def __len__(self): 
        return len(self.paths)
        
    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            return torch.zeros((3, 224, 224))

@torch.no_grad()
def embed_image_list(paths: List[str], clip_model: CLIPModel, processor: CLIPImageProcessor,
                     batch_size: int, device: str, amp: bool, amp_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    ds = ImageListDataset(paths, processor)
    # Removed num_workers and persistent_workers logic
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    chunks = []
    for batch in tqdm(loader, desc="Embedding images"):
        pv = batch.to(device, non_blocking=True)
        if amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                feats = clip_model.get_image_features(pv)
        else:
            feats = clip_model.get_image_features(pv)
        feats = F.normalize(feats, dim=-1)
        chunks.append(feats.detach().cpu().float())
        
    if not chunks: 
        return torch.tensor([])
        
    Z = torch.cat(chunks, dim=0)
    assert Z.shape[0] == len(paths)
    return Z

def build_groups_by_subdirs(root_dir: str) -> Tuple[List[str], List[List[str]]]:
    subdirs = find_prompt_subdirs(root_dir)
    if len(subdirs) == 0: 
        return [], []
    keys, groups = [], []
    for d in subdirs:
        imgs = list_sorted_images(d)
        if len(imgs) == 0: continue
        keys.append(os.path.basename(d))
        groups.append(imgs)
    return keys, groups

def build_groups_by_fixed_shots(root_dir: str, shots: int) -> Tuple[List[str], List[List[str]]]:
    imgs = list_sorted_images(root_dir)
    if len(imgs) == 0: 
        return [], []
    m = len(imgs) // shots
    if m == 0: 
        return ["chunk0"], [imgs]
    keys, groups = [], []
    for i in range(m):
        s, e = i * shots, (i + 1) * shots
        keys.append(f"chunk{i:05d}")
        groups.append(imgs[s:e])
    return keys, groups

def auto_build_groups(root_dir: str, shots: int) -> Tuple[List[str], List[List[str]], str]:
    keys, groups = build_groups_by_subdirs(root_dir)
    if len(keys) > 0: 
        return keys, groups, "subdir"
    keys, groups = build_groups_by_fixed_shots(root_dir, shots)
    if len(keys) > 0:
        return keys, groups, "fixed"
    return [], [], "none"

def compute_delta_per_prompt_schemeD(pos_groups, neg_groups,
                                     Z_pos_all, paths_pos_all,
                                     Z_neg_all, paths_neg_all,
                                     device) -> torch.Tensor:
    idx_pos: Dict[str, int] = {p: i for i, p in enumerate(paths_pos_all)}
    idx_neg: Dict[str, int] = {p: i for i, p in enumerate(paths_neg_all)}
    deltas = []
    used = 0
    
    for gpos, gneg in zip(pos_groups, neg_groups):
        pos_idx = [idx_pos[p] for p in gpos if p in idx_pos]
        neg_idx = [idx_neg[p] for p in gneg if p in idx_neg]
        if len(pos_idx) == 0 or len(neg_idx) == 0: 
            continue
            
        zpos = Z_pos_all[pos_idx]
        zneg = Z_neg_all[neg_idx]
        mpos = zpos.mean(dim=0)
        mneg = zneg.mean(dim=0)
        
        d = mpos - mneg
        n = torch.norm(d)
        if float(n) < 1e-8: 
            continue
            
        deltas.append((d / n).to(torch.float32))
        used += 1
        
    if used == 0:
        raise RuntimeError("Failed to construct any Delta, please check groups/data.")
        
    D = torch.stack(deltas, dim=0).to(device)
    print(f"[Delta] Scheme D generated {D.shape[0]} Delta vectors per group.")
    return D

def compute_kmeans_centers(feats: torch.Tensor, K: int, seed: int, name: str = "KMeans") -> torch.Tensor:
    n = feats.shape[0]
    if n < K:
        print(f"[{name}] Sample size {n} is less than K={K}. Adjusting K to {n}.")
        K = n
        
    print(f"[{name}] Performing K-Means clustering on {n} embeddings (K={K})...")
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=256, n_init=10, max_iter=200)
    kmeans.fit(feats.detach().cpu().numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    centers = F.normalize(centers, dim=-1)
    return centers