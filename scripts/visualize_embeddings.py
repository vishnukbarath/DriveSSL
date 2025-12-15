import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE

from src.models.resnet_simclr import get_resnet
from src.datasets.bdd_linear import BDDTimeOfDayDataset
from src.utils.device import get_device

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"
BATCH_SIZE = 64
MAX_SAMPLES = 2000
DEVICE = get_device()

LABEL_NAMES = {
    0: "Day",
    1: "Night",
    2: "Dawn/Dusk"
}

# ------------------------------
# MAIN
# ------------------------------
def main():
    print("[INFO] Loading dataset...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_dir = os.path.join(
        DATA_DIR,
        "raw",
        "bdd100k_original",
        "bdd100k",
        "bdd100k",
        "images",
        "100k",
        "val"
    )

    label_json = os.path.join(
        DATA_DIR,
        "raw",
        "bdd100k_original",
        "bdd100k_labels_release",
        "bdd100k",
        "labels",
        "bdd100k_labels_images_val.json"
    )

    dataset = BDDTimeOfDayDataset(img_dir, label_json, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("[INFO] Loading encoder...")
    encoder = get_resnet().to(DEVICE)
    encoder.eval()

    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)
    state_dict = ckpt.get("model_state", ckpt)

    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            fixed["backbone." + k[len("encoder."):]] = v

    encoder.load_state_dict(fixed, strict=False)

    features = []
    labels = []

    print("[INFO] Extracting features...")
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            feats = encoder(imgs)

            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

            if len(np.concatenate(features)) >= MAX_SAMPLES:
                break

    X = np.concatenate(features)[:MAX_SAMPLES]
    y = np.concatenate(labels)[:MAX_SAMPLES]

    print(f"[INFO] Extracted {X.shape[0]} samples with {X.shape[1]}-D features")

    print("[INFO] Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )

    X_2d = tsne.fit_transform(X)

    print("[INFO] Plotting...")
    plt.figure(figsize=(10, 8))

    for cls in np.unique(y):
        idx = y == cls
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=LABEL_NAMES[int(cls)],
            alpha=0.6,
            s=15
        )

    plt.legend()
    plt.title("t-SNE of SimCLR Learned Representations (BDD Time-of-Day)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/simclr_tsne.png", dpi=300)
    plt.show()

    print("[DONE] Visualization saved to visualizations/simclr_tsne.png")


if __name__ == "__main__":
    main()
