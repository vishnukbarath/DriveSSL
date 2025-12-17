import time
import torch
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.models.multihead_model import MultiHeadSimCLR

# ---------------- CONFIG ----------------
TRAIN_IMG = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\bdd100k\images\100k\train"
VAL_IMG   = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\bdd100k\images\100k\val"

TRAIN_JSON = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
VAL_JSON   = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
# --------------------------------------


def accuracy(pred, target):
    return (pred.argmax(1) == target).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    aw = asce = at = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)

        outputs = model(imgs)

        aw += accuracy(outputs["weather"], targets["weather"].to(device))
        asce += accuracy(outputs["scene"], targets["scene"].to(device))
        at += accuracy(outputs["timeofday"], targets["timeofday"].to(device))

    n = len(loader)
    return aw / n, asce / n, at / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # -------- DATASETS --------
    train_set = BDDMultiHeadDataset(TRAIN_IMG, TRAIN_JSON)
    val_set   = BDDMultiHeadDataset(VAL_IMG, VAL_JSON)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # -------- MODEL --------
    model = MultiHeadSimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # -------- METRICS --------
    train_loss = []
    train_acc, val_acc = [], []

    best_val = 0.0

    # -------- TRAIN LOOP --------
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_loss = 0.0
        aw = asce = at = 0.0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)

            outputs = model(imgs)

            lw = loss_fn(outputs["weather"], targets["weather"].to(device))
            ls = loss_fn(outputs["scene"], targets["scene"].to(device))
            lt = loss_fn(outputs["timeofday"], targets["timeofday"].to(device))

            loss = lw + ls + lt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            aw += accuracy(outputs["weather"], targets["weather"].to(device))
            asce += accuracy(outputs["scene"], targets["scene"].to(device))
            at += accuracy(outputs["timeofday"], targets["timeofday"].to(device))

        n = len(train_loader)
        avg_loss = total_loss / n
        aw /= n
        asce /= n
        at /= n

        train_loss.append(avg_loss)
        train_acc.append((aw + asce + at) / 3)

        # -------- VALIDATION --------
        v_aw, v_asce, v_at = evaluate(model, val_loader, device)
        val_avg = (v_aw + v_asce + v_at) / 3
        val_acc.append(val_avg)

        elapsed = time.time() - start
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.3f} | "
            f"Train Acc: {train_acc[-1]*100:.2f}% | "
            f"Val Acc: {val_avg*100:.2f}% | "
            f"Time: {elapsed:.1f}s | "
            f"GPU: {gpu_mem:.2f} GB | "
            f"CPU: {psutil.cpu_percent()}%"
        )

        if val_avg > best_val:
            best_val = val_avg
            torch.save(model.state_dict(), "bdd_multihead_best_val.pth")
            print(f"[INFO] Saved best validation model ({best_val*100:.2f}%)")

    print("[DONE] Training + Validation finished")

    # -------- FINAL PLOTS --------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.title("Average Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
