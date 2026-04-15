"""
EDA for Solar Panel Fault Dataset
================================
✔ Works with raw OR split dataset
✔ Shows class distribution
✔ Displays sample images
✔ Checks image sizes
✔ Detects imbalance
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter


DATA_DIR = "Faulty_solar_panels_split"   
SAMPLE_PER_CLASS = 5

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def get_images(root):
    class_images = {}

    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)

        if not os.path.isdir(cls_path):
            continue

        images = []
        for r, _, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(IMG_EXTENSIONS):
                    images.append(os.path.join(r, f))

        if len(images) > 0:
            class_images[cls] = images

    return class_images


data = get_images(DATA_DIR)

print("\n Classes found:", list(data.keys()))

counts = {cls: len(imgs) for cls, imgs in data.items()}

print("\n Class Distribution:")
for cls, cnt in counts.items():
    print(f"{cls:20s} : {cnt}")

plt.figure(figsize=(10,5))
plt.bar(counts.keys(), counts.values())
plt.xticks(rotation=30, ha="right")
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.show()

print("\n🖼 Showing sample images...")

for cls, imgs in data.items():
    samples = random.sample(imgs, min(SAMPLE_PER_CLASS, len(imgs)))

    plt.figure(figsize=(10,2))
    for i, img_path in enumerate(samples):
        img = Image.open(img_path)

        plt.subplot(1, SAMPLE_PER_CLASS, i+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")

    plt.suptitle(f"Samples: {cls}")
    plt.show()

print("\n Checking image sizes...")

sizes = []

for cls, imgs in data.items():
    for img_path in imgs[:100]:   
        try:
            img = Image.open(img_path)
            sizes.append(img.size)  
        except:
            continue

widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

print(f"Avg Width : {np.mean(widths):.2f}")
print(f"Avg Height: {np.mean(heights):.2f}")

plt.figure(figsize=(6,5))
plt.scatter(widths, heights, alpha=0.5)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.show()

# ─────────────────────────────
print("\n Checking imbalance...")

max_class = max(counts.values())
min_class = min(counts.values())

imbalance_ratio = max_class / min_class

print(f"Max samples: {max_class}")
print(f"Min samples: {min_class}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 2:
    print(" Dataset is IMBALANCED (use WeightedRandomSampler)")
else:
    print(" Dataset is fairly balanced")

# ─────────────────────────────
print("\n Random dataset preview")

all_images = []
for imgs in data.values():
    all_images.extend(imgs)

samples = random.sample(all_images, 9)

plt.figure(figsize=(6,6))
for i, img_path in enumerate(samples):
    img = Image.open(img_path)

    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.axis("off")

plt.suptitle("Random Images")
plt.show()

print("\n EDA Complete!")