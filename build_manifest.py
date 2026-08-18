#!/usr/bin/env python3
"""
Builds a train/val manifest for classification fine-tuning, combining Grocery
and Electronics into one 185-way subcategory classifier.

Splits by PRODUCT, not by image — all images of the same product stay together
in either train or val, so val never leaks a near-duplicate image of a training
product. Since most subcategories only have 3-5 products, we hold out exactly
1 product per class for val (classes with only 1 product go entirely to train,
since there's nothing to hold out).

Label = "Category/SubCategory" (kept distinct across the two taxonomies even
though no name collisions currently exist, as a safety margin).

Output: manifest.csv with columns: image_path, label, product_id, split
"""
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ("Grocery_Products.csv", "Grocery"),
    ("Electronics_Products.csv", "Electronics"),
]
OUT_PATH = Path(__file__).resolve().parent / "manifest.csv"


def main():
    products_by_label = defaultdict(list)  # label -> list of (product_id, [image_paths])

    for csv_name, folder_name in SOURCES:
        csv_path = ROOT / csv_name
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                subcat = row["SubCategory"].strip()
                label = f"{row['Category'].strip()}/{subcat}"
                folder = ROOT / folder_name / subcat.replace("/", "-").replace(":", "-")
                imgs = []
                for slot in range(1, 6):
                    p = row.get(f"Image_{slot}_Path", "").strip()
                    if p and (folder / p).exists():
                        imgs.append(str(folder / p))
                if not imgs:
                    continue
                products_by_label[label].append((row["ProductID"], imgs))

    rows_out = []
    n_train_products = n_val_products = 0
    single_product_classes = []
    for label, products in products_by_label.items():
        # Per-class RNG seeded from the label itself, not shared global state —
        # so one class's product count/order never perturbs another class's
        # train/val split (was a real bug: adding products to class A used to
        # shift which product got held out for unrelated class B).
        seed = int(hashlib.md5(label.encode()).hexdigest(), 16) % (2**32)
        random.Random(seed).shuffle(products)
        if len(products) >= 2:
            val_products = products[:1]
            train_products = products[1:]
        else:
            val_products = []
            train_products = products
            single_product_classes.append(label)

        for pid, imgs in train_products:
            n_train_products += 1
            for img in imgs:
                rows_out.append((img, label, pid, "train"))
        for pid, imgs in val_products:
            n_val_products += 1
            for img in imgs:
                rows_out.append((img, label, pid, "val"))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "product_id", "split"])
        writer.writerows(rows_out)

    n_train_imgs = sum(1 for r in rows_out if r[3] == "train")
    n_val_imgs = sum(1 for r in rows_out if r[3] == "val")
    print(f"Classes: {len(products_by_label)}")
    print(f"Products: {n_train_products} train, {n_val_products} val")
    print(f"Images: {n_train_imgs} train, {n_val_imgs} val")
    print(f"Classes with only 1 product (no val held out): {len(single_product_classes)}")
    for c in single_product_classes:
        print(f"   {c}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
