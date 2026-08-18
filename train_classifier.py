#!/usr/bin/env python3
"""
Trains and evaluates two classifiers on top of frozen SigLIP2 embeddings
(from embeddings.npz), and reports which one to keep:

1. Nearest-prototype: mean-embedding-per-class from train, classify by cosine
   similarity to nearest prototype. No training, robust to tiny per-class counts.
2. Linear probe: a single linear layer + softmax trained on train embeddings.

Both are evaluated on the held-out val split (whole products never seen during
"training" of either classifier, per build_manifest.py's per-product split).

Saves the better classifier + label mapping to checkpoints/ for later inference.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

ROOT = Path(__file__).resolve().parent
EMB_PATH = ROOT / "embeddings.npz"
CKPT_DIR = ROOT / "checkpoints"


def load_data():
    data = np.load(EMB_PATH, allow_pickle=True)
    return (data["embeddings"], data["labels"], data["splits"],
            data["product_ids"], data["paths"], str(data["model_name"][0]))


def prototype_classifier(X_train, y_train, X_val, y_val, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    prototypes = np.zeros((len(classes), X_train.shape[1]), dtype=np.float32)
    for c in classes:
        mask = y_train == c
        proto = X_train[mask].mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        prototypes[class_to_idx[c]] = proto

    sims = X_val @ prototypes.T  # cosine sim since both are L2-normalized
    preds_idx = sims.argmax(axis=1)
    preds = np.array([classes[i] for i in preds_idx])

    top1 = accuracy_score(y_val, preds)
    y_val_idx = np.array([class_to_idx[c] for c in y_val])
    top5 = top_k_accuracy_score(y_val_idx, sims, k=5, labels=np.arange(len(classes)))
    return top1, top5, prototypes, preds


def linear_probe(X_train, y_train, X_val, y_val, classes):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    top1 = accuracy_score(y_val, preds)
    probs = clf.predict_proba(X_val)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    y_val_idx = np.array([class_to_idx[c] for c in y_val])
    top5 = top_k_accuracy_score(y_val_idx, probs, k=5, labels=np.arange(len(clf.classes_)))
    return top1, top5, clf, preds


def main():
    X, y, splits, product_ids, paths, model_name = load_data()
    classes = sorted(set(y.tolist()))
    print(f"Embeddings: {X.shape}, model: {model_name}")
    print(f"Classes: {len(classes)}")

    train_mask = splits == "train"
    val_mask = splits == "val"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train images: {len(X_train)}, Val images: {len(X_val)} "
          f"(val products: {len(set(product_ids[val_mask].tolist()))})")

    print("\n=== Nearest-Prototype Classifier ===")
    proto_top1, proto_top5, prototypes, proto_preds = prototype_classifier(X_train, y_train, X_val, y_val, classes)
    print(f"Top-1 accuracy: {proto_top1:.4f}   Top-5 accuracy: {proto_top5:.4f}")

    print("\n=== Linear Probe (Logistic Regression) ===")
    lr_top1, lr_top5, clf, lr_preds = linear_probe(X_train, y_train, X_val, y_val, classes)
    print(f"Top-1 accuracy: {lr_top1:.4f}   Top-5 accuracy: {lr_top5:.4f}")

    CKPT_DIR.mkdir(exist_ok=True)

    if lr_top1 >= proto_top1:
        print(f"\n>>> Linear probe wins ({lr_top1:.4f} vs {proto_top1:.4f}) — saving it as the classifier.")
        import pickle
        with open(CKPT_DIR / "classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
        chosen, preds = "linear_probe", lr_preds
    else:
        print(f"\n>>> Prototype classifier wins ({proto_top1:.4f} vs {lr_top1:.4f}) — saving prototypes.")
        np.savez(CKPT_DIR / "prototypes.npz", prototypes=prototypes, classes=np.array(classes))
        chosen, preds = "prototype", proto_preds

    with open(CKPT_DIR / "labels.json", "w") as f:
        json.dump({"classes": classes, "model_name": model_name, "chosen_classifier": chosen}, f, indent=2)

    report = classification_report(y_val, preds, zero_division=0)
    with open(CKPT_DIR / "val_report.txt", "w") as f:
        f.write(f"Chosen classifier: {chosen}\n")
        f.write(f"Nearest-prototype: top1={proto_top1:.4f} top5={proto_top5:.4f}\n")
        f.write(f"Linear probe:      top1={lr_top1:.4f} top5={lr_top5:.4f}\n\n")
        f.write(report)
    print(f"\nSaved classifier + label map + per-class report to {CKPT_DIR}/")

    # quick summary of worst-performing classes (useful for spotting label-noise candidates)
    from collections import defaultdict
    correct = defaultdict(int)
    total = defaultdict(int)
    for true, pred in zip(y_val, preds):
        total[true] += 1
        if true == pred:
            correct[true] += 1
    worst = sorted(total.keys(), key=lambda c: correct[c] / total[c])[:15]
    print("\nWorst-performing classes on val (possible label-noise or ambiguous-class candidates):")
    for c in worst:
        print(f"  {correct[c]}/{total[c]}  {c}")


if __name__ == "__main__":
    main()
