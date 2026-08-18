#!/usr/bin/env python3
"""
Classifies a new product image using the trained checkpoint (frozen SigLIP2
backbone + the classifier chosen by train_classifier.py).

Usage:
  python3 classify.py path/to/image.jpg
  python3 classify.py path/to/image.jpg --top 5
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_image(image_path, model, processor, device):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        if not torch.is_tensor(feats):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    with open(CKPT_DIR / "labels.json") as f:
        meta = json.load(f)
    model_name = meta["model_name"]
    chosen = meta["chosen_classifier"]

    device = get_device()
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    vec = embed_image(args.image_path, model, processor, device)

    if chosen == "linear_probe":
        with open(CKPT_DIR / "classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        scores = clf.predict_proba(vec.reshape(1, -1))[0]
        classes = clf.classes_
        score_label = "probability"
    else:
        data = np.load(CKPT_DIR / "prototypes.npz", allow_pickle=True)
        prototypes, classes = data["prototypes"], data["classes"]
        scores = prototypes @ vec  # raw cosine similarity, not softmax — see confidence_analysis.py
        score_label = "similarity"

    order = np.argsort(-scores)[: args.top]
    print(f"\nTop {args.top} predictions for {args.image_path}:")
    for rank, i in enumerate(order, 1):
        print(f"  {rank}. {classes[i]:<45} {scores[i]*100:5.1f}% {score_label}")
    if chosen != "linear_probe" and scores[order[0]] < 0.70:
        print("\n  Low confidence (<70% similarity) — treat as a rough suggestion, not a final answer.")


if __name__ == "__main__":
    main()
