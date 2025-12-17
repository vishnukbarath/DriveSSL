import time
import psutil
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.models.multihead_model import MultiHeadSimCLR

# ---------- CONFIG ----------
IMAGE_ROOT = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\bdd100k\images\100k\train"
LABEL_JSON = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
# ----------------------------


def accuracy(pred, target):
    return (pred.argmax(1) == target).float().mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    dataset = BDDMultiHeadDataset(IMAGE_ROOT, LABEL_JSON)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = MultiHeadSimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # -------- METRICS STORAGE --------
    losses = []
    acc_weather = []
    acc_scene = []
    acc_time = []

    best_acc = 0.0

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_loss = 0.0
        aw = asce = at = 0.0

        for imgs, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)

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

        n = len(loader)
        epoch_loss = total_loss / n
        aw /= n
        asce /= n
        at /= n

        losses.append(epoch_loss)
        acc_weather.append(aw)
        acc_scene.append(asce)
        acc_time.append(at)

        elapsed = time.time() - start
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {epoch_loss:.3f} | "
            f"Weather: {aw*100:.2f}% | "
            f"Scene: {asce*100:.2f}% | "
            f"Time: {at*100:.2f}% | "
            f"Epoch Time: {elapsed:.1f}s | "
            f"GPU: {gpu_mem:.2f} GB | "
            f"CPU: {psutil.cpu_percent()}%"
        )

        avg_acc = (aw + asce + at) / 3
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "bdd_multihead_best.pth")
            print(f"[INFO] Saved best model ({best_acc*100:.2f}%)")

    print("[DONE] Training complete")

    # ---------- FINAL VISUALIZATION ----------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(losses, marker="o")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(acc_weather, label="Weather")
    axs[1].plot(acc_scene, label="Scene")
    axs[1].plot(acc_time, label="Time-of-Day")
    axs[1].set_title("Training Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
