import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.models.multihead_model import MultiHeadSimCLR
from src.utils.device import get_device

# =========================================================
# PATHS
# =========================================================
IMAGE_ROOT = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\bdd100k\images\100k\train"
LABEL_JSON = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"

MODEL_PATH = "experiments/multihead/checkpoints/best_model.pth"
SAVE_DIR = "experiments/multihead/confusion"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LABEL MAPS (MUST MATCH DATASET ENCODING)
# =========================================================
TIME_MAP = {0: "daytime", 1: "night", 2: "dawn/dusk"}
WEATHER_MAP = {0: "clear", 1: "overcast", 2: "rainy", 3: "foggy", 4: "snowy"}
SCENE_MAP = {0: "city street", 1: "highway", 2: "residential", 3: "parking lot"}

# =========================================================
def evaluate():
    device = get_device()
    print(f"[INFO] Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BDDMultiHeadDataset(
        img_dir=IMAGE_ROOT,
        label_json=LABEL_JSON,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    model = MultiHeadSimCLR().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # -----------------------------------------------------
    y_true = {"time": [], "weather": [], "scene": []}
    y_pred = {"time": [], "weather": [], "scene": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs = batch["image"].to(device)

            out = model(imgs)

            for key in ["time", "weather", "scene"]:
                y_true[key].extend(batch[key].numpy())
                y_pred[key].extend(out[key].argmax(1).cpu().numpy())

    # -----------------------------------------------------
    plot_cm("time", y_true["time"], y_pred["time"], TIME_MAP)
    plot_cm("weather", y_true["weather"], y_pred["weather"], WEATHER_MAP)
    plot_cm("scene", y_true["scene"], y_pred["scene"], SCENE_MAP)

    print(f"[DONE] Confusion matrices saved to {SAVE_DIR}")

# =========================================================
def plot_cm(name, y_true, y_pred, label_map):
    labels = list(label_map.keys())
    names = list(label_map.values())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name.upper()} Confusion Matrix")

    path = os.path.join(SAVE_DIR, f"{name}_confusion.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {path}")

# =========================================================
if __name__ == "__main__":
    evaluate()


