#!/usr/bin/env python3
"""
Extracts frozen SigLIP2 image embeddings for every image in manifest.csv and
caches them to embeddings.npz. This is the expensive step (one forward pass
per image through the ViT) — run once, then train/eval scripts just load the
cached vectors instead of re-running the backbone.

Resumable: saves a checkpoint every CHECKPOINT_EVERY images. If interrupted
(closed terminal, killed process, etc.), just re-run the exact same command —
it picks up where it left off instead of starting over.

Usage:
  python3 extract_embeddings.py [--model google/siglip2-base-patch16-224] [--batch-size 16]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.csv"
OUT_PATH = ROOT / "embeddings.npz"
CHECKPOINT_PATH = ROOT / "embeddings_checkpoint.npz"
CHECKPOINT_EVERY = 500  # images; a run this long needs to survive interruptions


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading {args.model} ...")
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} images to embed")

    all_embeds = []
    all_labels = []
    all_splits = []
    all_product_ids = []
    all_paths = []
    done_paths = set()

    if CHECKPOINT_PATH.exists():
        ckpt = np.load(CHECKPOINT_PATH, allow_pickle=True)
        if str(ckpt["model_name"][0]) == args.model:
            all_embeds = list(ckpt["embeddings"])
            all_labels = list(ckpt["labels"])
            all_splits = list(ckpt["splits"])
            all_product_ids = list(ckpt["product_ids"])
            all_paths = list(ckpt["paths"])
            done_paths = set(all_paths)
            print(f"Resuming from checkpoint: {len(done_paths)} images already embedded "
                  f"(interrupted run picks up where it left off)")
        else:
            print(f"Checkpoint found but for a different model ({ckpt['model_name'][0]!r}) — starting fresh")

    rows = [r for r in rows if r["image_path"] not in done_paths]
    print(f"{len(rows)} remaining to embed")

    def save_checkpoint():
        np.savez(
            CHECKPOINT_PATH,
            embeddings=np.stack(all_embeds), labels=np.array(all_labels),
            splits=np.array(all_splits), product_ids=np.array(all_product_ids),
            paths=np.array(all_paths), model_name=np.array([args.model]),
        )

    t0 = time.time()
    since_checkpoint = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        images = []
        keep = []
        for r in batch:
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                images.append(img)
                keep.append(r)
            except Exception as e:
                print(f"  [SKIP] {r['image_path']}: {e}")
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if not torch.is_tensor(feats):
                feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize for cosine similarity
        feats = feats.cpu().numpy()

        for r, vec in zip(keep, feats):
            all_embeds.append(vec)
            all_labels.append(r["label"])
            all_splits.append(r["split"])
            all_product_ids.append(r["product_id"])
            all_paths.append(r["image_path"])

        done = start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{len(rows)} this run  ({rate:.1f} img/s)", flush=True)

        since_checkpoint += len(keep)
        if since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint()
            since_checkpoint = 0
            print(f"  [checkpoint saved: {len(all_embeds)} images total]", flush=True)

    embeds = np.stack(all_embeds)
    np.savez(
        OUT_PATH,
        embeddings=embeds,
        labels=np.array(all_labels),
        splits=np.array(all_splits),
        product_ids=np.array(all_product_ids),
        paths=np.array(all_paths),
        model_name=np.array([args.model]),
    )
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f"\nSaved {embeds.shape} embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()
