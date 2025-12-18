import os
import time
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.models.multihead_model import MultiHeadSimCLR

# ===================== CONFIG =====================
IMAGE_ROOT = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\bdd100k\images\100k\train"
LABEL_JSON = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 4

EXP_DIR = "experiments/multihead"
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
METRIC_DIR = os.path.join(EXP_DIR, "metrics")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
# =================================================


def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return p, r, f


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    dataset = BDDMultiHeadDataset(IMAGE_ROOT, LABEL_JSON)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = MultiHeadSimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --------- LOG STORAGE ----------
    history = {
        "loss": [],
        "weather_acc": [],
        "scene_acc": [],
        "time_acc": [],
        "weather_f1": [],
        "scene_f1": [],
        "time_f1": [],
    }

    best_avg_f1 = 0.0

    print(f"[INFO] Starting training for {EPOCHS} epochs")

    for epoch in range(EPOCHS):
        model.train()
        start = time.time()

        total_loss = 0.0

        gt_w, pr_w = [], []
        gt_s, pr_s = [], []
        gt_t, pr_t = [], []

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

        for imgs, targets in loop:
            imgs = imgs.to(device)

            out = model(imgs)

            lw = loss_fn(out["weather"], targets["weather"].to(device))
            ls = loss_fn(out["scene"], targets["scene"].to(device))
            lt = loss_fn(out["timeofday"], targets["timeofday"].to(device))

            loss = lw + ls + lt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Predictions
            pw = out["weather"].argmax(1).cpu().numpy()
            ps = out["scene"].argmax(1).cpu().numpy()
            pt = out["timeofday"].argmax(1).cpu().numpy()

            gt_w.extend(targets["weather"].numpy())
            gt_s.extend(targets["scene"].numpy())
            gt_t.extend(targets["timeofday"].numpy())

            pr_w.extend(pw)
            pr_s.extend(ps)
            pr_t.extend(pt)

            loop.set_postfix(loss=loss.item())

        # -------- METRICS ----------
        epoch_loss = total_loss / len(loader)

        w_acc = np.mean(np.array(gt_w) == np.array(pr_w))
        s_acc = np.mean(np.array(gt_s) == np.array(pr_s))
        t_acc = np.mean(np.array(gt_t) == np.array(pr_t))

        _, _, w_f1 = compute_metrics(gt_w, pr_w)
        _, _, s_f1 = compute_metrics(gt_s, pr_s)
        _, _, t_f1 = compute_metrics(gt_t, pr_t)

        history["loss"].append(epoch_loss)
        history["weather_acc"].append(w_acc)
        history["scene_acc"].append(s_acc)
        history["time_acc"].append(t_acc)
        history["weather_f1"].append(w_f1)
        history["scene_f1"].append(s_f1)
        history["time_f1"].append(t_f1)

        avg_f1 = (w_f1 + s_f1 + t_f1) / 3

        elapsed = time.time() - start
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0

        print(
            f"\n[Epoch {epoch+1}] "
            f"Loss: {epoch_loss:.3f} | "
            f"W Acc: {w_acc*100:.2f}% | "
            f"S Acc: {s_acc*100:.2f}% | "
            f"T Acc: {t_acc*100:.2f}% | "
            f"Avg F1: {avg_f1*100:.2f}% | "
            f"Time: {elapsed:.1f}s | "
            f"GPU: {gpu_mem:.2f} GB"
        )

        # -------- SAVE BEST MODEL ----------
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            torch.save(
                model.state_dict(),
                os.path.join(CKPT_DIR, "best_model.pth")
            )
            print(f"[INFO] Saved new best model (Avg F1 {best_avg_f1*100:.2f}%)")

    # ---------- SAVE METRICS ----------
    csv_path = os.path.join(METRIC_DIR, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))

    # ---------- FINAL PLOTS ----------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(history["weather_f1"], label="Weather")
    plt.plot(history["scene_f1"], label="Scene")
    plt.plot(history["time_f1"], label="Time")
    plt.legend()
    plt.title("F1 Score")

    plt.tight_layout()
    plt.show()

    print("[DONE] Training + metrics completed")


if __name__ == "__main__":
    main()
