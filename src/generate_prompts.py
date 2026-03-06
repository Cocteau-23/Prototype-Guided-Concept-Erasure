import argparse
import csv
import json
import os
import random
import re
from typing import Dict, List, Tuple, Any, Optional

def _choice(lst: List[str], p_drop: float = 0.0) -> str:
    if not lst or (p_drop > 0 and random.random() < p_drop): 
        return ""
    return random.choice(lst)

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,", ", ", s)
    return s.strip(" ,")

class _SafeDict(dict):
    def __missing__(self, key): 
        return ""

def load_schema(default_path: str = "schema.json", custom_path: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.isfile(default_path):
        raise FileNotFoundError(f"Default schema not found: {default_path}")
        
    with open(default_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if not custom_path:
        return schema
        
    if not os.path.isfile(custom_path):
        raise FileNotFoundError(f"Custom schema not found: {custom_path}")
        
    with open(custom_path, "r", encoding="utf-8") as f:
        user_s = json.load(f)
        
    for k, v in user_s.get("global", {}).items():
        if isinstance(v, list) and isinstance(schema["global"].get(k), list):
            schema["global"][k] = list(dict.fromkeys(schema["global"][k] + v))
        else:
            schema["global"][k] = v
            
    for cat, cfg in user_s.get("categories", {}).items():
        if cat not in schema["categories"]:
            schema["categories"][cat] = cfg
            continue
        for k, v in cfg.items():
            if isinstance(v, list) and isinstance(schema["categories"][cat].get(k), list):
                schema["categories"][cat][k] = list(dict.fromkeys(schema["categories"][cat][k] + v))
            elif isinstance(v, dict) and isinstance(schema["categories"][cat].get(k), dict):
                schema["categories"][cat][k].update(v)
            else:
                schema["categories"][cat][k] = v
                
    return schema

def _compose_from_facets(c: Dict[str, Any], near_miss: bool) -> Tuple[str, Dict[str, str]]:
    facets_key = "near_miss_facets" if near_miss else "concept_facets"
    tmpls_key  = "near_miss_templates" if near_miss else "concept_templates"
    
    facets = c.get(facets_key, {})
    if not facets: 
        return "", {}

    picked = {k: random.choice(lst) for k, lst in facets.items() if lst}

    templates = c.get(tmpls_key, [])
    if templates:
        phrase = random.choice(templates).format_map(_SafeDict(picked))
    else:
        phrase = ", ".join(filter(None, picked.values()))
    
    return _clean(phrase), picked

def _make_row(prompt: str, category: str, parts: Dict[str, Any], proto_hint: str, seed: int, 
              near_miss: int = 0, pair_id: str = "", variant: str = "", 
              concept_facets: dict = None) -> Dict[str, Any]:
    row = {
        "prompt": _clean(prompt), 
        "negative_prompt": "", 
        "category": category,
        "proto_hint": proto_hint, 
        "seed": seed, 
        "near_miss": near_miss,
        "pair_id": pair_id, 
        "variant": variant, 
        "concept_facets": json.dumps(concept_facets or {}, ensure_ascii=False)
    }
    row.update(parts)
    return row

def generate_pairs(cat: str, schema: Dict[str, Any], n: int, dropout: float) -> List[Dict[str, Any]]:
    g = schema["global"]
    c = schema["categories"][cat]
    rows = []
    
    for i in range(n):
        parts_base = {
            "style": _choice(g.get("styles", []), dropout), 
            "subject": _choice(g.get("subjects", []), dropout),
            "scene": _choice(g.get("scenes", []), dropout), 
            "lighting": _choice(g.get("lighting", []), dropout),
            "camera": _choice(g.get("cameras", []), dropout), 
            "mods": _choice(g.get("modifiers", []), dropout)
        }
        
        con_on, meta_on = _compose_from_facets(c, near_miss=False)
        con_off, meta_off = _compose_from_facets(c, near_miss=True)
        
        tpl = random.choice(c.get("templates", ["{concept}"]))
        parts_on = {**parts_base, "concept": con_on}
        parts_off = {**parts_base, "concept": con_off}
        
        p_on = tpl.format_map(_SafeDict(parts_on))
        p_off = tpl.format_map(_SafeDict(parts_off))
        
        seed = random.randint(0, 2**31 - 1)
        pair_id = f"{cat}#{i:06d}"

        rows.append(_make_row(p_on, cat, parts_on, f"{cat}#concept:{con_on}", seed, 
                              near_miss=0, pair_id=pair_id, variant="on", concept_facets=meta_on))
        rows.append(_make_row(p_off, f"{cat}_off", parts_off, f"{cat}#near_miss:{con_off or 'blank'}", seed, 
                              near_miss=1, pair_id=pair_id, variant="off", concept_facets=meta_off))
    return rows

def main():
    ap = argparse.ArgumentParser(description="Generate paired prompts for Text-to-Image models.")
    ap.add_argument("--out", type=str, default="data/test.csv", help="Output CSV path")
    ap.add_argument("--categories", type=str, required=True, help="Comma-separated list of categories")
    ap.add_argument("--n_per_category", type=int, default=100, help="Number of pairs per category")
    ap.add_argument("--dropout", type=float, default=0.0, help="Probability to drop optional global modifiers")
    ap.add_argument("--seed", type=int, default=100, help="Random seed for reproducibility")
    ap.add_argument("--schema", type=str, default=None, help="Path to custom schema JSON to merge")
    ap.add_argument("--schema_default", type=str, default="schema.json", help="Path to default schema JSON")
    ap.add_argument("--no_dedup", action="store_true", help="Disable prompt deduplication")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    schema = load_schema(default_path=args.schema_default, custom_path=args.schema)
    
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    for c in cats:
        if c not in schema["categories"]: 
            raise ValueError(f"Unknown category: '{c}'")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for cat in cats:
        rows.extend(generate_pairs(cat, schema, args.n_per_category, args.dropout))

    # Deduplicate prompts
    if not args.no_dedup:
        seen = set()
        deduped = []
        for r in rows:
            k = (r["prompt"].lower(), r["category"], r.get("variant",""))
            if k not in seen:
                seen.add(k)
                deduped.append(r)
        rows = deduped

    fields = ["prompt", "negative_prompt", "category", "proto_hint", "seed", "style", "subject",
              "scene", "lighting", "camera", "mods", "concept", "near_miss", "pair_id", "variant", "concept_facets"]
    
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: 
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"[Done] Generated {len(rows)} prompts -> {args.out}")
    
    from collections import Counter
    stats = ", ".join(f"{k}:{v}" for k, v in Counter(r["category"] for r in rows).most_common())
    print(f"[Stats] {stats}")

if __name__ == "__main__":
    main()