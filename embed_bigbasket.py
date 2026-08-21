#!/usr/bin/env python3
"""
Embeds the BigBasket catalog with the SAME backbone the deployed classifier
uses, so those products can be audited in bigbasket_eval_app.py without a live
forward pass per click.

Why this is a genuinely useful evaluation set: 839 BigBasket products were
merged into our training data earlier, but 1,170 were not. Those 1,170 are
genuinely held-out — real retail listings, photographed by someone else, that
the prototypes have never seen. This script records which is which so the app can
separate "unseen" from "already trained on"; only the unseen half is honest
evidence of generalization.

Two kinds of image are kept apart deliberately:
  product : Image_<slot>_<id>.jpg        - the actual product photograph
  brand   : Brand_Image_<slot>_<id>.jpg  - a brand logo/banner, NOT a product.
            A classifier trained on product photos has no sensible answer for
            a logo, so these are embedded but tagged, never mixed into the
            product numbers.

Resumable: checkpoints every CHECKPOINT_EVERY images. Re-run the identical
command after an interruption and it continues instead of restarting.

Usage:
  python3 embed_bigbasket.py                      # product images only (default)
  python3 embed_bigbasket.py --include-brand      # also embed brand logos
  python3 embed_bigbasket.py --unseen-only        # skip products already trained on
  python3 embed_bigbasket.py --batch-size 2       # if you hit OOM
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent                 # SigLIP2_Training
DATA_ROOT = ROOT.parent                                # Dataset for Training SigLIP
BB_ROOT = DATA_ROOT.parent / "Big Basket"
BB_CSV = BB_ROOT / "BigBasket.csv"
BB_IMAGES = BB_ROOT / "Big Basket Dataset"

CKPT_DIR = ROOT / "checkpoints"
MANIFEST = ROOT / "manifest.csv"
OUT_PATH = ROOT / "bigbasket_embeddings.npz"
CHECKPOINT_PATH = ROOT / "bigbasket_embeddings_checkpoint.npz"
CHECKPOINT_EVERY = 500

PRODUCT_RE = re.compile(r"^Image_(\d+)_(\d+)\.\w+$")
BRAND_RE = re.compile(r"^Brand_Image_(\d+)_(\d+)\.\w+$")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def trained_bigbasket_ids() -> set[str]:
    """BigBasket product ids that already went into our training set.

    Merged training images carry a BB prefix (Image_1_BB40193879.jpg) while the
    BigBasket source names them without it (Image_1_40193879.jpg)."""
    ids = set()
    if not MANIFEST.exists():
        return ids
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.match(r"^Image_\d+_BB(\d+)\.", Path(row["image_path"]).name)
            if m:
                ids.add(m.group(1))
    return ids


def collect_rows(include_brand: bool, unseen_only: bool):
    csv.field_size_limit(10 ** 9)
    trained = trained_bigbasket_ids()
    print(f"{len(trained)} BigBasket product ids are already in training")

    out = []
    seen_products = set()
    with open(BB_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Product id comes from the image filename, which is the only field
            # that reliably carries it.
            pid = None
            for slot in range(1, 6):
                m = PRODUCT_RE.match(row.get(f"Image_{slot}_Path", "").strip() or "")
                if m:
                    pid = m.group(2)
                    break
            if pid is None:
                continue
            already = pid in trained
            if unseen_only and already:
                continue

            meta = {
                "product_id": pid,
                "name": row.get("ProductName", "").strip(),
                "brand": row.get("Brand", "").strip(),
                "bb_category": row.get("Category", "").strip(),
                "bb_subcategory": row.get("SubCategory", "").strip(),
                "in_training": already,
            }
            kinds = [("product", f"Image_{{}}_Path")]
            if include_brand:
                kinds.append(("brand", f"Brand_Image_{{}}_Path"))
            for kind, tmpl in kinds:
                for slot in range(1, 6):
                    rel = row.get(tmpl.format(slot), "").strip()
                    if not rel:
                        continue
                    p = BB_IMAGES / rel
                    if p.exists():
                        out.append({**meta, "kind": kind, "path": str(p)})
            seen_products.add(pid)
    print(f"{len(seen_products)} products, {len(out)} images to embed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-brand", action="store_true",
                    help="also embed Brand_Image_* logos (tagged separately)")
    ap.add_argument("--unseen-only", action="store_true",
                    help="only products NOT already in our training set")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not BB_CSV.exists():
        sys.exit(f"BigBasket.csv not found at {BB_CSV}")

    with open(CKPT_DIR / "labels.json") as f:
        model_name = json.load(f)["model_name"]
    print(f"Backbone (matching deployed model): {model_name}")

    rows = collect_rows(args.include_brand, args.unseen_only)
    if not rows:
        sys.exit("nothing to embed")

    device = get_device()
    print(f"Device: {device}")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    embeds, meta_keep = [], []
    done = set()
    if CHECKPOINT_PATH.exists():
        ck = np.load(CHECKPOINT_PATH, allow_pickle=True)
        if str(ck["model_name"][0]) == model_name:
            embeds = list(ck["embeddings"])
            meta_keep = list(ck["meta"])
            done = {m["path"] for m in meta_keep}
            print(f"Resuming: {len(done)} images already embedded")
        else:
            print("Checkpoint is for a different backbone — starting fresh")

    todo = [r for r in rows if r["path"] not in done]
    print(f"{len(todo)} remaining")

    def save(path):
        np.savez(path, embeddings=np.stack(embeds) if embeds else np.zeros((0, 1)),
                 meta=np.array(meta_keep, dtype=object),
                 model_name=np.array([model_name]))

    t0 = time.time()
    since = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        imgs, keep = [], []
        for r in batch:
            try:
                imgs.append(Image.open(r["path"]).convert("RGB"))
                keep.append(r)
            except Exception as e:
                print(f"  [SKIP] {r['path']}: {e}")
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if not torch.is_tensor(feats):
                feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for r, v in zip(keep, feats.cpu().numpy()):
            embeds.append(v)
            meta_keep.append(r)

        done_n = start + len(batch)
        rate = done_n / max(time.time() - t0, 1e-9)
        eta = (len(todo) - done_n) / max(rate, 1e-9) / 60
        print(f"  {done_n}/{len(todo)}  ({rate:.1f} img/s, ~{eta:.0f} min left)", flush=True)

        since += len(keep)
        if since >= CHECKPOINT_EVERY:
            save(CHECKPOINT_PATH)
            since = 0
            print(f"  [checkpoint: {len(embeds)} total]", flush=True)

    save(OUT_PATH)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f"\nSaved {len(embeds)} embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()
