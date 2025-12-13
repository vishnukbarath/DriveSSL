import os
import shutil
from tqdm import tqdm

# ===== PATHS =====
RAW_ROOT = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original"
IMAGE_ROOT = os.path.join(
    RAW_ROOT,
    "bdd100k",
    "bdd100k",
    "images",
    "100k"
)

OUTPUT_DIR = r"C:\Users\vishn\Documents\DriveSSL\data\ssl_unlabeled"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def collect_images(split_dir):
    image_paths = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                image_paths.append(os.path.join(root, f))
    return image_paths

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] Collecting images for SSL pretraining...")

    train_dir = os.path.join(IMAGE_ROOT, "train")
    val_dir = os.path.join(IMAGE_ROOT, "val")

    images = []
    images.extend(collect_images(train_dir))
    images.extend(collect_images(val_dir))

    print(f"[INFO] Total images found: {len(images)}")

    for idx, img_path in enumerate(tqdm(images)):
        new_name = f"img_{idx:07d}.jpg"
        dst_path = os.path.join(OUTPUT_DIR, new_name)
        shutil.copy2(img_path, dst_path)

    print(f"[DONE] SSL dataset ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
