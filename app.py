#!/usr/bin/env python3
"""
Streamlit frontend for the Grocery + Electronics SigLIP2 subcategory classifier.

Upload a product image; it embeds the image with the frozen SigLIP2 backbone
and compares it against the per-subcategory prototype vectors (mean embedding
of each class's training images), showing top-3 and top-5 predictions side by
side with confidence scores so you can see whether they agree.

Run with:
  streamlit run app.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"

# "Datasets/Electronics and Grocery Dataset/Combined Dataset" — the raw scraped
# product catalog (Ground Truth.csv + downloaded images), two levels above
# "Dataset for Training SigLIP/SigLIP2_Training".
COMBINED_ROOT = ROOT.parent.parent / "Electronics and Grocery Dataset" / "Combined Dataset"
COMBINED_CONFIG = {
    "Electronics": {
        "csv": COMBINED_ROOT / "Electronics" / "Ground Truth.csv",
        "images_dir": COMBINED_ROOT / "Electronics" / "Electronics Dataset",
        "name_col": "Item Name",
    },
    "Grocery": {
        "csv": COMBINED_ROOT / "Grocery" / "Ground Truth.csv",
        "images_dir": COMBINED_ROOT / "Grocery" / "Grocery Dataset",
        "name_col": "Product Name",
    },
}
IMAGE_FILENAME_COLS = [
    "Image 1 Filename", "Image 2 Filename", "Image 3 Filename",
    "Image 4 / Video 1 Filename", "Image 5 / Video 2 Filename",
]


@st.cache_data(show_spinner="Indexing product catalog...")
def load_products(category):
    """Rows from Combined Dataset's Ground Truth.csv that have at least one
    image file actually present on disk (most rows are metadata-only)."""
    cfg = COMBINED_CONFIG[category]
    if not cfg["csv"].is_file():
        return []
    products = []
    with open(cfg["csv"], newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            images = [
                fn for col in IMAGE_FILENAME_COLS
                if (fn := (row.get(col) or "").strip()) and (cfg["images_dir"] / fn).is_file()
            ]
            if not images:
                continue
            name = (row.get(cfg["name_col"]) or "").strip() or f"(unnamed row {i})"
            brand = (row.get("Brand") or "").strip()
            label = f"{name} — {brand}" if brand and brand.lower() not in name.lower() else name
            products.append({"label": label, "images": images})
    return products


@st.cache_resource(show_spinner="Loading SigLIP2 model (first run only)...")
def load_model_and_classifier():
    with open(CKPT_DIR / "labels.json") as f:
        meta = json.load(f)
    model_name = meta["model_name"]

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    proto_data = np.load(CKPT_DIR / "prototypes.npz", allow_pickle=True)
    prototypes, classes = proto_data["prototypes"], proto_data["classes"]

    return model, processor, device, prototypes, classes, model_name, meta


def embed_image(image, model, processor, device):
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        if not torch.is_tensor(feats):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def confidence_badge(sim):
    # Buckets calibrated against confidence_analysis.py's measured accuracy-by-threshold
    # table on the held-out val set, not an arbitrary scale.
    if sim >= 0.85:
        return "🟢 High confidence"
    if sim >= 0.70:
        return "🟡 Medium confidence"
    return "🔴 Low confidence — needs review"


def render_predictions(classes, sims, order, k):
    st.subheader(f"Top {k}")
    for rank in range(k):
        i = order[rank]
        label = classes[i]
        category, subcat = label.split("/", 1)
        pct = max(sims[i], 0) * 100
        st.write(f"**{rank + 1}. {subcat}**  _(​{category})_  —  {confidence_badge(sims[i])}")
        st.progress(min(int(pct), 100), text=f"{pct:.1f}% similarity to this class")


def main():
    st.set_page_config(page_title="Product Subcategory Classifier", page_icon="📦", layout="wide")
    st.title("📦 Grocery + Electronics Subcategory Classifier")
    st.caption("Fine-tuned-free classification on top of frozen SigLIP2 embeddings — "
               "185 subcategories across the Grocery and Electronics taxonomies.")

    model, processor, device, prototypes, classes, model_name, meta = load_model_and_classifier()

    with st.sidebar:
        st.markdown("### Model info")
        st.write(f"**Backbone:** `{model_name}`")
        st.write(f"**Classifier:** {meta['chosen_classifier']}")
        st.write(f"**Classes:** {len(classes)}")
        st.write(f"**Device:** {device}")

    source = st.radio(
        "Image source", ["Upload your own image", "Browse product catalog"],
        horizontal=True,
    )

    image = None
    image_caption = "Uploaded image"

    if source == "Upload your own image":
        uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

    else:
        sel_col1, sel_col2 = st.columns([1, 3])
        with sel_col1:
            category = st.selectbox("Category", list(COMBINED_CONFIG.keys()))

        products = load_products(category)
        with sel_col2:
            product = (
                st.selectbox("Product", products, format_func=lambda p: p["label"])
                if products else None
            )
        if not products:
            st.info(f"No {category} rows with a locally downloaded image were found.")

        if product:
            image_name = (
                st.selectbox("Image", product["images"]) if len(product["images"]) > 1
                else product["images"][0]
            )
            image_path = COMBINED_CONFIG[category]["images_dir"] / image_name
            image = Image.open(image_path).convert("RGB")
            image_caption = f"{product['label']}  ({image_name})"

    if image is not None:
        col_img, col_results = st.columns([1, 2])

        with col_img:
            st.image(image, caption=image_caption, use_container_width=True)

        with st.spinner("Classifying..."):
            vec = embed_image(image, model, processor, device)
            sims = prototypes @ vec  # cosine similarity, both sides L2-normalized
            order = np.argsort(-sims)

        with col_results:
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                render_predictions(classes, sims, order, 3)
            with res_col2:
                render_predictions(classes, sims, order, 5)

            best_sim = sims[order[0]]
            if best_sim < 0.70:
                st.warning(
                    f"⚠️ Low confidence ({best_sim*100:.1f}% similarity) — the correct answer may "
                    "not even be in this list. This usually means the photo doesn't closely "
                    "resemble the clean product photos the model was trained on (e.g. a lifestyle/"
                    "ad-banner shot instead of a plain product image), or the true subcategory is a "
                    "close visual neighbor of ones shown here. Treat this as a rough suggestion, "
                    "not a final answer."
                )


if __name__ == "__main__":
    main()
