import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet_simclr import get_resnet
from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.utils.device import get_device

# ---------------- CONFIG ---------------- #

DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
IMG_TRAIN = os.path.join(
    DATA_DIR,
    "raw",
    "bdd100k_original",
    "bdd100k",
    "bdd100k",
    "images",
    "100k",
    "train",
)
LABEL_TRAIN = os.path.join(
    DATA_DIR,
    "raw",
    "bdd100k_original",
    "bdd100k_labels_release",
    "bdd100k",
    "labels",
    "bdd100k_labels_images_train.json",
)

BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4

NUM_TIME = 3
NUM_WEATHER = 6
NUM_DOMAIN = 2

SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"

DEVICE = get_device()

# ---------------- MODEL ---------------- #

class MultiHeadSimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_resnet()

        self.head_time = nn.Linear(2048, NUM_TIME)
        self.head_weather = nn.Linear(2048, NUM_WEATHER)
        self.head_domain = nn.Linear(2048, NUM_DOMAIN)

    def forward(self, x):
        feats = self.encoder(x)
        return {
            "time": self.head_time(feats),
            "weather": self.head_weather(feats),
            "domain": self.head_domain(feats),
        }


# ---------------- MAIN ---------------- #

def main():
    print(f"[INFO] Using device: {DEVICE}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = BDDMultiHeadDataset(
        IMG_TRAIN,
        LABEL_TRAIN,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = MultiHeadSimCLR().to(DEVICE)

    # ---- Load SimCLR encoder ---- #
    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)

    filtered = {
        k.replace("encoder.", "backbone."): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }

    model.encoder.load_state_dict(filtered, strict=False)
    print("[INFO] Loaded SimCLR encoder weights")

    # ---- Losses & Optimizer ---- #
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("[INFO] Starting multi-head training")

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_loss = 0.0

        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            time_y = batch["time"].to(DEVICE)
            weather_y = batch["weather"].to(DEVICE)
            domain_y = batch["domain"].to(DEVICE)

            out = model(imgs)

            loss_time = loss_fn(out["time"], time_y)
            loss_weather = loss_fn(out["weather"], weather_y)
            loss_domain = loss_fn(out["domain"], domain_y)

            loss = loss_time + loss_weather + loss_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {total_loss/len(loader):.4f} "
            f"Time: {time.time()-start:.2f}s"
        )

    os.makedirs("experiments/multihead", exist_ok=True)
    torch.save(
        model.state_dict(),
        "experiments/multihead/multihead_model.pth",
    )

    print("[DONE] Multi-head model saved")


# ---------------- ENTRY ---------------- #

if __name__ == "__main__":
    main()
