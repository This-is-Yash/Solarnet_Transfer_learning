"""
ADVANCED EDA — Solar Panel Fault Dataset
=======================================
✔ Deep analysis (for report / viva)
✔ Multiple graphs
✔ Dataset quality insights
"""

import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

DATA_DIR = "Faulty_solar_panels_split"  

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

data = defaultdict(list)

for cls in os.listdir(DATA_DIR):
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    for root, _, files in os.walk(cls_path):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                data[cls].append(os.path.join(root, f))

classes = list(data.keys())
counts = [len(data[c]) for c in classes]

print("\nClasses:", classes)

plt.figure()
plt.bar(classes, counts)
plt.xticks(rotation=30)
plt.title("Class Distribution")
plt.ylabel("Images")
plt.show()

plt.figure()
plt.pie(counts, labels=classes, autopct='%1.1f%%')
plt.title("Class Distribution (Pie)")
plt.show()

widths, heights, ratios = [], [], []

for cls in data:
    for img_path in data[cls][:100]:  
        try:
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            ratios.append(w/h)
        except:
            continue

plt.figure()
plt.hist(widths, bins=30)
plt.title("Width Distribution")
plt.show()

plt.figure()
plt.hist(heights, bins=30)
plt.title("Height Distribution")
plt.show()

plt.figure()
plt.scatter(widths, heights, alpha=0.5)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Scatter")
plt.show()

plt.figure()
plt.hist(ratios, bins=30)
plt.title("Aspect Ratio Distribution")
plt.show()

r_vals, g_vals, b_vals = [], [], []

for cls in data:
    for img_path in data[cls][:50]:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            r_vals.extend(img[:,:,0].flatten())
            g_vals.extend(img[:,:,1].flatten())
            b_vals.extend(img[:,:,2].flatten())
        except:
            continue

plt.figure()
plt.hist(r_vals, bins=50, alpha=0.5)
plt.title("Red Channel Distribution")
plt.show()

plt.figure()
plt.hist(g_vals, bins=50, alpha=0.5)
plt.title("Green Channel Distribution")
plt.show()

plt.figure()
plt.hist(b_vals, bins=50, alpha=0.5)
plt.title("Blue Channel Distribution")
plt.show()

brightness = []

for cls in data:
    for img_path in data[cls][:100]:
        try:
            img = np.array(Image.open(img_path).convert("L"))
            brightness.append(np.mean(img))
        except:
            continue

plt.figure()
plt.hist(brightness, bins=30)
plt.title("Brightness Distribution")
plt.show()

print("\nClass-wise stats:")
for cls in classes:
    print(f"{cls:20s} → {len(data[cls])}")

imbalance_ratio = max(counts) / min(counts)
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")

plt.figure(figsize=(6,6))

all_imgs = []
for cls in data:
    all_imgs.extend(data[cls])

samples = random.sample(all_imgs, 9)

for i, path in enumerate(samples):
    img = Image.open(path)
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.axis("off")

plt.suptitle("Random Samples")
plt.show()

print("\n Advanced EDA Completed!")