#!/usr/bin/env python3
"""
Evaluate the trained classifier against the BigBasket catalog — an external
retail dataset with its own taxonomy, photographed by someone else.

Products are grouped by their filename id prefix (every Image_<slot>_<id>.jpg
sharing an <id> is one product), shown with BigBasket's own Category /
SubCategory / Brand beside the model's top predictions.

Two things this deliberately does NOT do:

  * It does not report an "accuracy". BigBasket's 335 subcategories are a
    different taxonomy from our 185 classes and there is no faithful 1:1 map,
    so any accuracy number would be measuring the mapping, not the model. This
    is a judgement tool: you look and decide whether the prediction is right.

  * It does not mix in brand logos. Brand_Image_* files are logos/banners, not
    product photographs — a product classifier has no meaningful answer for
    them. They are viewable under a separate toggle, tagged as such.

839 of these products were merged into our training set earlier; 1,170 were
not. The sidebar defaults to the UNSEEN ones, because only those are honest
evidence of generalization. Products already trained on are marked.

Run with:
  cd "Dataset for Training SigLIP/SigLIP2_Training"
  streamlit run bigbasket_eval_app.py --server.port 8504
"""
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
EMB_PATH = ROOT / "bigbasket_embeddings.npz"
CKPT_PARTIAL = ROOT / "bigbasket_embeddings_checkpoint.npz"

PRODUCT_COLS = 5
TOP_K = 3

st.set_page_config(page_title="BigBasket Evaluation", layout="wide")


def confidence_badge(sim: float) -> str:
    if sim >= 0.85:
        return "🟢"
    if sim >= 0.70:
        return "🟡"
    return "🔴"


@st.cache_resource(show_spinner="Loading BigBasket embeddings…")
def load_data():
    """Falls back to the in-progress checkpoint so the app is usable while
    embed_bigbasket.py is still running."""
    src = EMB_PATH if EMB_PATH.exists() else (CKPT_PARTIAL if CKPT_PARTIAL.exists() else None)
    if src is None or not (CKPT_DIR / "prototypes.npz").exists():
        return None

    emb = np.load(src, allow_pickle=True)
    proto = np.load(CKPT_DIR / "prototypes.npz", allow_pickle=True)
    X = emb["embeddings"]
    meta = list(emb["meta"])
    if len(X) == 0:
        return None

    prototypes, classes = proto["prototypes"], proto["classes"]
    sims = X @ prototypes.T
    order = np.argsort(-sims, axis=1)[:, :TOP_K]

    # group image rows into products
    products = defaultdict(lambda: {"product": [], "brand": []})
    for i, m in enumerate(meta):
        preds = [(str(classes[j]), float(sims[i, j])) for j in order[i]]
        entry = {**m, "preds": preds}
        products[m["product_id"]][m["kind"]].append(entry)

    return {
        "products": dict(products),
        "model_name": str(emb["model_name"][0]),
        "n_images": len(X),
        "partial": src is CKPT_PARTIAL,
    }


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


data = load_data()

st.title("BigBasket Evaluation — external catalog, unseen products")

if data is None:
    st.error(
        "No BigBasket embeddings yet. Run:\n\n"
        "```\npython3 embed_bigbasket.py --include-brand\n```\n\n"
        "It takes roughly 100 minutes for the full catalog and checkpoints as it "
        "goes — this page becomes usable from the first checkpoint onward.")
    st.stop()

products = data["products"]
if data["partial"]:
    st.warning(f"Reading an **in-progress** embedding run ({data['n_images']} images so far). "
               "Re-run this page later for the full catalog.")
st.caption(f"Predictions from `{data['model_name']}` · {data['n_images']} BigBasket images "
           f"· {len(products)} products")

# ---- sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("Filter")

    scope = st.radio(
        "Which products",
        ["Unseen only (true held-out)", "Already in training", "All"],
        key="scope",
        help="839 BigBasket products were merged into our training set. Only the "
             "unseen ones are honest evidence of generalization.")

    def prod_meta(pid):
        rows = products[pid]["product"] or products[pid]["brand"]
        return rows[0] if rows else None

    pids = [p for p in products if prod_meta(p)]
    if scope.startswith("Unseen"):
        pids = [p for p in pids if not prod_meta(p)["in_training"]]
    elif scope.startswith("Already"):
        pids = [p for p in pids if prod_meta(p)["in_training"]]

    cats = sorted({prod_meta(p)["bb_category"] for p in pids if prod_meta(p)["bb_category"]})
    cat = st.selectbox("BigBasket category", ["(all)"] + cats, key="cat")
    if cat != "(all)":
        pids = [p for p in pids if prod_meta(p)["bb_category"] == cat]

    subs = sorted({prod_meta(p)["bb_subcategory"] for p in pids
                   if prod_meta(p)["bb_subcategory"]})
    sub = st.selectbox("BigBasket subcategory", ["(all)"] + subs, key="sub")
    if sub != "(all)":
        pids = [p for p in pids if prod_meta(p)["bb_subcategory"] == sub]

    brands = sorted({prod_meta(p)["brand"] for p in pids if prod_meta(p)["brand"]})
    brand = st.selectbox("Brand", ["(all)"] + brands, key="brand")
    if brand != "(all)":
        pids = [p for p in pids if prod_meta(p)["brand"] == brand]

    st.divider()
    low_conf = st.checkbox("Only low-confidence (<70%)", key="lowconf",
                           help="Where the model is unsure — the cases worth reading.")
    show_brand = st.checkbox("Show brand logo images", key="showbrand",
                             help="Brand_Image_* files are logos, not products. "
                                  "The model has no meaningful answer for these.")
    limit = st.slider("Products to show", 10, 300, 60, step=10, key="limit")

    st.divider()
    st.caption(f"**{len(pids)}** products match this filter")

pids = sorted(pids)

# ---- headline distribution ------------------------------------------------
all_top = []
for p in pids:
    for r in products[p]["product"]:
        all_top.append(r["preds"][0])
if all_top:
    conf = np.array([s for _, s in all_top])
    c1, c2, c3 = st.columns(3)
    c1.metric("Product images shown", len(all_top))
    c2.metric("Median confidence", f"{np.median(conf)*100:.0f}%")
    c3.metric("High confidence (≥85%)", f"{(conf >= 0.85).mean()*100:.0f}%")

    pred_counts = defaultdict(int)
    for lbl, _ in all_top:
        pred_counts[lbl] += 1
    top_preds = sorted(pred_counts.items(), key=lambda kv: -kv[1])[:12]
    with st.expander("What our model maps this selection onto (top predicted classes)"):
        for lbl, n in top_preds:
            st.write(f"`{n:5d}`  {lbl}   ({n/len(all_top)*100:.0f}%)")

st.divider()

# ---- product cards --------------------------------------------------------
shown = 0
for pid in pids:
    if shown >= limit:
        break
    bucket = products[pid]
    rows = list(bucket["product"])
    if show_brand:
        rows += bucket["brand"]
    if not rows:
        continue
    if low_conf and not any(r["preds"][0][1] < 0.70 for r in rows if r["kind"] == "product"):
        continue

    m = (bucket["product"] or bucket["brand"])[0]
    shown += 1

    flag = " · :orange[in training set]" if m["in_training"] else ""
    st.markdown(f"**{m['name'][:100] or pid}**  ·  `{pid}`{flag}")
    st.caption(f"BigBasket: {m['bb_category']} → {m['bb_subcategory']}"
               + (f"  ·  Brand: {m['brand']}" if m["brand"] else ""))

    for chunk_start in range(0, len(rows), PRODUCT_COLS):
        cols = st.columns(PRODUCT_COLS)   # fixed count so every tile is one size
        for col, r in zip(cols, rows[chunk_start:chunk_start + PRODUCT_COLS]):
            with col:
                p = Path(r["path"])
                try:
                    st.image(square_thumb(r["path"], p.stat().st_mtime),
                             use_container_width=True)
                except Exception:
                    st.markdown("*(unreadable)*")
                if r["kind"] == "brand":
                    st.markdown(
                        "<div style='font-size:11px;color:#8a6d3b'>brand logo — "
                        "not a product</div>", unsafe_allow_html=True)
                lines = []
                for rank, (lbl, s) in enumerate(r["preds"], 1):
                    short = lbl.split("/", 1)[-1]
                    color = "#1a7f37" if rank == 1 and s >= 0.70 else (
                        "#c0392b" if rank == 1 else "#777")
                    weight = "600" if rank == 1 else "400"
                    tick = confidence_badge(s) if rank == 1 else ""
                    lines.append(
                        f"<div style='font-size:11px;color:{color};font-weight:{weight}'>"
                        f"{tick} {short} · {s*100:.0f}%</div>")
                st.markdown("".join(lines), unsafe_allow_html=True)
    st.divider()

if shown == 0:
    st.info("No products match this filter.")
