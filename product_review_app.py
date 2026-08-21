#!/usr/bin/env python3
"""
Per-product model-audit frontend: browse a subcategory, see every product's
images grouped together (all Image_1..Image_5 that share a ProductID), with
the trained model's predicted subcategory laid right next to the ground-truth
label from the CSV.

Fast by construction: it does NOT run the model live. extract_embeddings.py
already embedded every image once; this just reloads that cache (embeddings.npz)
and the deployed prototypes (checkpoints/prototypes.npz) and does one matrix
multiply for the whole dataset — a few hundred milliseconds for a giant-opt
run over 15k images, not a live forward pass per image. No torch/transformers
needed at all here.

Purpose: hunt for mislabeled ground truth and hard/ambiguous products by
sorting straight to what the model disagrees with, grouped the way a human
actually reviews a listing — by product, not by lone image.

Run with:
  cd "Dataset for Training SigLIP/SigLIP2_Training"
  streamlit run product_review_app.py     # opens on :8503
"""
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent          # SigLIP2_Training
DATA_ROOT = ROOT.parent                          # Dataset for Training SigLIP
CKPT_DIR = ROOT / "checkpoints"
EMB_PATH = ROOT / "embeddings.npz"

SOURCES = [("Electronics_Products.csv", "Electronics"), ("Grocery_Products.csv", "Grocery")]
IMG_SLOTS = [f"Image_{i}_Path" for i in range(1, 6)]
PRODUCT_COLS = 5   # fixed column count for every product card — see comment below

st.set_page_config(page_title="Model Audit", layout="wide")
st.markdown(
    "<style>div[data-testid='stCheckbox']{min-height:38px;display:flex;"
    "align-items:center}</style>", unsafe_allow_html=True)


def confidence_badge(sim: float) -> str:
    if sim >= 0.85:
        return "🟢"
    if sim >= 0.70:
        return "🟡"
    return "🔴"


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading cached embeddings + prototypes…")
def load_predictions():
    """One matrix multiply over the whole dataset — no model, no GPU forward
    pass. Returns {abs_path_str: (pred_label, confidence)} plus metadata."""
    if not EMB_PATH.exists() or not (CKPT_DIR / "prototypes.npz").exists():
        return None
    emb = np.load(EMB_PATH, allow_pickle=True)
    proto = np.load(CKPT_DIR / "prototypes.npz", allow_pickle=True)

    X, paths = emb["embeddings"], emb["paths"]
    prototypes, classes = proto["prototypes"], proto["classes"]
    model_name = str(emb["model_name"][0])

    sims = X @ prototypes.T                      # (N images) x (185 classes)
    best = sims.argmax(axis=1)
    pred_map = {
        str(p): (str(classes[best[i]]), float(sims[i, best[i]]))
        for i, p in enumerate(paths)
    }
    return {"pred": pred_map, "model_name": model_name, "n_embedded": len(paths)}


@st.cache_data(show_spinner="Reading product catalog…")
def load_catalog():
    """category -> subcategory -> [ {product_id, name, images:[abs_path,...]} ]"""
    tree: dict[str, dict[str, list[dict]]] = {}
    for csv_name, category in SOURCES:
        p = DATA_ROOT / csv_name
        if not p.exists():
            continue
        tree.setdefault(category, defaultdict(list))
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                subcat = row["SubCategory"].strip()
                folder = DATA_ROOT / category / subcat.replace("/", "-").replace(":", "-")
                imgs = []
                for slot in IMG_SLOTS:
                    rel = row.get(slot, "").strip()
                    if rel and (folder / rel).exists():
                        imgs.append(str(folder / rel))
                if imgs:
                    tree[category][subcat].append({
                        "product_id": row["ProductID"], "name": row.get("ProductName", ""),
                        "images": imgs,
                    })
    return {c: dict(subs) for c, subs in tree.items()}


@st.cache_data(show_spinner=False, max_entries=8000)
def square_thumb(path: str, _mtime: float, size: int = 220) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=85)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
preds = load_predictions()
catalog = load_catalog()

st.title("Model Audit — predictions vs. ground truth, grouped by product")

if preds is None:
    st.error(
        "No embeddings.npz / checkpoints/prototypes.npz found. Run the training "
        "pipeline first (extract_embeddings.py, then train_classifier.py) — "
        "see commands.txt.")
    st.stop()

pred_map = preds["pred"]
st.caption(f"Predictions from `{preds['model_name']}` · {preds['n_embedded']} images embedded "
           "· computed once from the cached run, not live inference")
st.caption(
    ":gray[Accuracy shown here includes images the prototypes were built from, so it reads "
    "higher than the model's real generalization accuracy — use `confidence_analysis.py` / "
    "`checkpoints/val_report.txt` for the honest held-out number. This view is for **finding "
    "disagreements to investigate**, not for measuring the model.]")

with st.sidebar:
    st.header("Navigate")
    category = st.radio("Category", sorted(catalog), horizontal=True, key="category")
    subcats = sorted(catalog.get(category, {}))

    # Per-subcategory accuracy, so you can jump straight to the worst ones.
    def subcat_acc(cat, sub):
        correct = total = 0
        for prod in catalog[cat][sub]:
            for img in prod["images"]:
                pv = pred_map.get(img)
                if pv is None:
                    continue
                total += 1
                if pv[0] == f"{cat}/{sub}":
                    correct += 1
        return correct, total

    stats = {s: subcat_acc(category, s) for s in subcats}

    sort_mode = st.radio("Sort subcategories by", ["Name", "Worst accuracy first"],
                         key="sortmode")
    if sort_mode == "Worst accuracy first":
        subcats = sorted(subcats, key=lambda s: (stats[s][0] / stats[s][1]
                                                  if stats[s][1] else 1.0))

    def fmt(s):
        c, t = stats[s]
        pct = f"{c/t*100:.0f}%" if t else "—"
        return f"{s}  ({pct}, {t} imgs)"

    subcat = st.radio("Subcategory", subcats, format_func=fmt,
                      key=f"subcat_{category}", label_visibility="collapsed")

    st.divider()
    only_mismatch = st.checkbox("Show only model disagreements", key="only_mismatch")
    st.caption("Flags where the model's top prediction differs from the "
               "CSV's SubCategory label — either a model miss or a mislabeled product.")

if not subcat:
    st.info("No subcategories found.")
    st.stop()

# ---- main -----------------------------------------------------------------
products = catalog[category][subcat]
truth_label = f"{category}/{subcat}"
c_correct, c_total = stats[subcat]

hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.subheader(f"{category} / {subcat}")
    st.caption(f"{len(products)} products · {c_total} images embedded")
with hdr_r:
    if c_total:
        acc = c_correct / c_total
        st.metric("Model accuracy here", f"{acc*100:.0f}%", f"{c_correct}/{c_total} correct")

st.divider()

shown_products = 0
for prod in products:
    rows = []
    for img in prod["images"]:
        pv = pred_map.get(img)
        if pv is None:
            continue
        pred_label, sim = pv
        rows.append((img, pred_label, sim, pred_label == truth_label))
    if not rows:
        continue
    if only_mismatch and all(r[3] for r in rows):
        continue

    shown_products += 1
    n_wrong = sum(1 for r in rows if not r[3])
    title = f"**{prod['name'][:90] or prod['product_id']}**  ·  `{prod['product_id']}`"
    if n_wrong:
        title += f"  ·  :red[{n_wrong}/{len(rows)} images disagree]"
    st.markdown(title)

    # Always allocate the SAME number of columns regardless of how many images
    # this product has. Sizing columns to len(rows) was the bug: a 2-image
    # product got wide ~50%-width columns while a 6-image product got narrow
    # ~16%-width columns, so the identical 220x220 thumbnail rendered at wildly
    # different on-screen sizes from product to product. Unused slots in a
    # short row are just left empty, which keeps every image the same size.
    cols = st.columns(PRODUCT_COLS)
    for col, (img, pred_label, sim, ok) in zip(cols, rows):
        img_path = Path(img)
        with col:
            try:
                st.image(square_thumb(img, img_path.stat().st_mtime), use_container_width=True)
            except Exception:
                st.markdown("*(unreadable)*")
            badge = confidence_badge(sim)
            if ok:
                st.markdown(f"<div style='font-size:12px;color:#1a7f37'>"
                            f"✅ {badge} {sim*100:.0f}%</div>", unsafe_allow_html=True)
            else:
                pred_sub = pred_label.split("/", 1)[-1]
                st.markdown(
                    f"<div style='font-size:12px;color:#c0392b'>"
                    f"❌ {badge} {sim*100:.0f}%<br>predicted: {pred_sub}</div>",
                    unsafe_allow_html=True)
    st.divider()

if shown_products == 0:
    st.success("No model disagreements in this subcategory." if only_mismatch
               else "No embedded images to show for this subcategory.")
