import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.multihead_model import MultiHeadSimCLR
from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.utils.device import get_device

# ---------------- CONFIG ---------------- #
DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
IMG_TRAIN = os.path.join(DATA_DIR, "raw", "bdd100k_original",
                         "bdd100k", "bdd100k", "images", "100k", "train")
LABEL_TRAIN = os.path.join(DATA_DIR, "raw", "bdd100k_original",
                           "bdd100k_labels_release", "bdd100k", "labels",
                           "bdd100k_labels_images_train.json")

SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"
SAVE_DIR = "experiments/multihead"

EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-4
DEVICE = get_device()

# ---------------------------------------- #

def main():
    print(f"[INFO] Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BDDMultiHeadDataset(IMG_TRAIN, LABEL_TRAIN, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4)

    model = MultiHeadSimCLR().to(DEVICE)

    # -------- Load SimCLR Encoder -------- #
    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)
    simclr_state = ckpt["model_state"]

    encoder_state = {}
    for k, v in simclr_state.items():
        if k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "encoder.backbone.")] = v

    model.load_state_dict(encoder_state, strict=False)
    print("[INFO] Loaded SimCLR encoder weights")

    # -------- Losses -------- #
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------- Training -------- #
    for epoch in range(EPOCHS):
        model.train()
        correct_t = correct_w = correct_d = total = 0
        loss_sum = 0

        start = time.time()

        for imgs, t, w, d in loader:
            imgs = imgs.to(DEVICE)
            t = t.to(DEVICE)
            w = w.to(DEVICE)
            d = d.to(DEVICE)

            out = model(imgs)

            loss = (
                ce(out["time"], t) +
                ce(out["weather"], w) +
                ce(out["domain"], d)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            correct_t += (out["time"].argmax(1) == t).sum().item()
            correct_w += (out["weather"].argmax(1) == w).sum().item()
            correct_d += (out["domain"].argmax(1) == d).sum().item()
            total += t.size(0)

        acc = (correct_t + correct_w + correct_d) / (3 * total) * 100
        elapsed = time.time() - start

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {loss_sum:.3f} "
              f"Avg Acc: {acc:.2f}% "
              f"Time: {elapsed:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_multihead.pth")
            print(f"[INFO] Saved best model ({best_acc:.2f}%)")

    print(f"[DONE] Best Avg Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
