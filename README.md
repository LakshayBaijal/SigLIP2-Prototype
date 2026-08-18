# SigLIP-Prototype

Product-image subcategory classifier for 185 subcategories (94 Electronics +
91 Grocery), built on a frozen [SigLIP2](https://huggingface.co/docs/transformers/model_doc/siglip2)
vision backbone with a nearest-prototype classifier on top (no fine-tuning of
the backbone — zero trainable parameters, just a per-class mean embedding).

## Dataset

The training dataset (product images, organized by category/subcategory) is
**not included in this repo** — it's several GB of images. It's hosted here:

**[Google Drive — SigLIP training dataset](https://drive.google.com/drive/folders/1ouAFpnTPljIspL6ISWrYLBneZS8tJRNx?usp=drive_link)**

Download it and place it alongside this folder so the relative paths in
`build_manifest.py` / `manifest.csv` resolve correctly (see that script for
the expected folder layout).

## How it works

1. **`build_manifest.py`** — splits the dataset into train/val by *product*
   (not by image), so multiple photos of the same product never leak across
   the split. Each class gets an independent random seed.
2. **`extract_embeddings.py`** — runs every image through the frozen SigLIP2
   backbone once and caches the resulting embeddings (`embeddings.npz`).
   Resumable — safe to interrupt and re-run.
3. **`train_classifier.py`** — builds per-class prototype vectors (mean
   embedding per subcategory) and evaluates nearest-prototype classification
   against a linear-probe baseline on the held-out val set.
4. **`confidence_analysis.py`** — measures accuracy as a function of
   cosine-similarity confidence threshold, to pick a sensible
   auto-accept/route-to-review cutoff for production use.
5. **`classify.py`** — classify a single image from the command line.
6. **`app.py`** — Streamlit demo: upload an image (or browse the product
   catalog) and see top-3/top-5 predicted subcategories with confidence.

## Setup

```bash
pip3 install torch transformers scikit-learn pillow streamlit
```

## Running the pipeline

```bash
python3 build_manifest.py
python3 extract_embeddings.py --model google/siglip2-so400m-patch14-384 --batch-size 8
python3 train_classifier.py
python3 confidence_analysis.py
python3 classify.py "/path/to/image.jpg"
streamlit run app.py
```

See `commands.txt` for the full command reference, including the giant-opt
backbone variant, the 25-trial repeated-holdout validation snippet, and
instructions for rebuilding deployed prototypes from the full dataset.

## Repo contents

- Training/eval scripts (`build_manifest.py`, `extract_embeddings.py`,
  `train_classifier.py`, `confidence_analysis.py`, `classify.py`)
- `app.py` — Streamlit frontend
- `checkpoints/` — the currently deployed prototype vectors + label map
  (small, ~1MB — this is the trained model, not the dataset)
- `commands.txt` — full command reference and notes

Not included: raw dataset images, cached embeddings (`embeddings.npz`),
and old backup checkpoints — all regenerable from the dataset above via the
pipeline steps.
