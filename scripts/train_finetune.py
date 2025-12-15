import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.resnet_simclr import get_resnet
from src.datasets.bdd_linear import BDDTimeOfDayDataset
from src.utils.device import get_device

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"


BATCH_SIZE = 64
NUM_EPOCHS = 25
ENCODER_LR = 1e-4      # lower LR for backbone
CLASSIFIER_LR = 1e-3  # higher LR for head
NUM_CLASSES = 3

SAVE_DIR = "lane_segmentation"
DEVICE = get_device()

# ------------------------------
# MAIN
# ------------------------------
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # ------------------------------
    # TRANSFORMS
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ------------------------------
    # DATASET
    # ------------------------------
    train_folder = os.path.join(DATA_DIR, "train")
    val_folder = os.path.join(DATA_DIR, "val")

    if os.path.isdir(train_folder) and os.listdir(train_folder):
        train_dataset = datasets.ImageFolder(train_folder, transform=transform)
        val_dataset = datasets.ImageFolder(val_folder, transform=transform)
    else:
        bdd_images_train = os.path.join(
            DATA_DIR, "raw", "bdd100k_original", "bdd100k",
            "bdd100k", "images", "100k", "train"
        )
        bdd_images_val = os.path.join(
            DATA_DIR, "raw", "bdd100k_original", "bdd100k",
            "bdd100k", "images", "100k", "val"
        )
        labels_dir = os.path.join(
            DATA_DIR, "raw", "bdd100k_original",
            "bdd100k_labels_release", "bdd100k", "labels"
        )

        train_json = os.path.join(labels_dir, "bdd100k_labels_images_train.json")
        val_json = os.path.join(labels_dir, "bdd100k_labels_images_val.json")

        train_dataset = BDDTimeOfDayDataset(bdd_images_train, train_json, transform)
        val_dataset = BDDTimeOfDayDataset(bdd_images_val, val_json, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ------------------------------
    # MODEL
    # ------------------------------
    encoder = get_resnet().to(DEVICE)
    classifier = nn.Linear(2048, NUM_CLASSES).to(DEVICE)

    # ------------------------------
    # LOAD SIMCLR CHECKPOINT
    # ------------------------------
    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)
    state_dict = ckpt.get("model_state", ckpt)

    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            mapped["backbone." + k[len("encoder."):]] = v

    encoder.load_state_dict(mapped, strict=False)
    print("[INFO] Loaded SimCLR encoder weights")

    # ------------------------------
    # OPTIMIZER
    # ------------------------------
    optimizer = optim.Adam([
        {"params": encoder.parameters(), "lr": ENCODER_LR},
        {"params": classifier.parameters(), "lr": CLASSIFIER_LR},
    ])

    criterion = nn.CrossEntropyLoss()

    # ------------------------------
    # TRAIN LOOP
    # ------------------------------
    best_val_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        encoder.train()
        classifier.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            feats = encoder(imgs)
            outputs = classifier(feats)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss /= len(train_dataset)

        # ------------------------------
        # VALIDATION
        # ------------------------------
        encoder.eval()
        classifier.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                feats = encoder(imgs)
                outputs = classifier(feats)

                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}% "
            f"Time: {elapsed:.2f}s"
        )

        # ------------------------------
        # SAVE BEST
        # ------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch + 1
            }, os.path.join(SAVE_DIR, "finetuned_best.pth"))
            print(f"[INFO] Saved best model (Val Acc: {val_acc:.2f}%)")

    print(f"[DONE] Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
