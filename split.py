

import os
import random
import shutil


DATA_DIR = "C:/Users/Yash/Downloads/archive/Faulty_solar_panel"          
SPLIT_DIR = "Faulty_solar_panels_split"  

VAL_SPLIT = 0.15
TEST_SPLIT = 0.05

SEED = 42
random.seed(SEED)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def split_dataset(src, dst):


    if os.path.exists(os.path.join(dst, "train")):
        print(" Split already exists. Skipping...")
        return

    print(" Creating dataset split...\n")

    total_images = 0

    for cls in os.listdir(src):
        cls_path = os.path.join(src, cls)

        if not os.path.isdir(cls_path):
            continue

        print(f"Processing class: {cls}")

        image_files = []
        for root, _, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(IMG_EXTENSIONS):
                    image_files.append(os.path.join(root, f))

        if len(image_files) == 0:
            print(f" No images found in {cls}, skipping...")
            continue
        random.shuffle(image_files)

        n = len(image_files)
        n_test = int(n * TEST_SPLIT)
        n_val  = int(n * VAL_SPLIT)

        splits = {
            "test": image_files[:n_test],
            "val": image_files[n_test:n_test+n_val],
            "train": image_files[n_test+n_val:]
        }

        for split_name, files in splits.items():
            split_path = os.path.join(dst, split_name, cls)
            os.makedirs(split_path, exist_ok=True)

            for file_path in files:
                filename = os.path.basename(file_path)

                shutil.copy(file_path, os.path.join(split_path, filename))

        print(f"  → {n} images split (Train/Val/Test)")

        total_images += n

    print("\n Dataset split completed!")
    print(f" Total images processed: {total_images}")
    print(f" Output directory: {dst}")


if __name__ == "__main__":
    split_dataset(DATA_DIR, SPLIT_DIR)