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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = MultiHeadSimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------- VIS DATA ----------
    losses = []
    acc_weather, acc_scene, acc_time = [], [], []

    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    best_acc = 0.0

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_loss = 0
        aw = asce = at = 0

        for imgs, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
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
            f"Time: {elapsed:.1f}s | "
            f"GPU: {gpu_mem:.2f} GB"
        )

        # ---------- SAVE BEST ----------
        avg_acc = (aw + asce + at) / 3
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "bdd_multihead_best.pth")
            print(f"[INFO] Saved best model ({best_acc*100:.2f}%)")

        # ---------- LIVE PLOTS ----------
        axs[0, 0].cla()
        axs[0, 0].plot(losses)
        axs[0, 0].set_title("Total Loss")

        axs[0, 1].cla()
        axs[0, 1].plot(acc_weather, label="Weather")
        axs[0, 1].plot(acc_scene, label="Scene")
        axs[0, 1].plot(acc_time, label="Time")
        axs[0, 1].legend()
        axs[0, 1].set_title("Accuracy")

        axs[1, 0].cla()
        axs[1, 0].bar(
            ["CPU %", "RAM %"],
            [psutil.cpu_percent(), psutil.virtual_memory().percent]
        )
        axs[1, 0].set_title("System Load")

        axs[1, 1].axis("off")

        plt.pause(0.1)

    plt.ioff()
    plt.show()
    print("[DONE] Training complete")

if __name__ == "__main__":
    main()
