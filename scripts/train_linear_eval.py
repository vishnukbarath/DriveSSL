import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.resnet_simclr import get_resnet  # Your ResNetSimCLR model
from src.datasets.bdd_linear import BDDTimeOfDayDataset
from src.utils.device import get_device

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3
# BDD time-of-day has 3 classes: daytime, night, dawn/dusk
NUM_CLASSES = 3
SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"
DEVICE = get_device()  # Should return 'cuda' if available

def main():
    # ------------------------------
    # DATASET & DATALOADER
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Prefer ImageFolder structure if present, otherwise try BDD time-of-day dataset
    train_folder = os.path.join(DATA_DIR, "train")
    val_folder = os.path.join(DATA_DIR, "val")
    if os.path.isdir(train_folder) and os.listdir(train_folder):
        train_dataset = datasets.ImageFolder(train_folder, transform=transform)
        val_dataset = datasets.ImageFolder(val_folder, transform=transform)
    else:
        # Look for BDD100k images and labels inside the raw data directory
        bdd_images_train = os.path.join(DATA_DIR, "raw", "bdd100k_original", "bdd100k", "bdd100k", "images", "100k", "train")
        bdd_images_val = os.path.join(DATA_DIR, "raw", "bdd100k_original", "bdd100k", "bdd100k", "images", "100k", "val")
        bdd_labels_dir = os.path.join(DATA_DIR, "raw", "bdd100k_original", "bdd100k_labels_release", "bdd100k", "labels")
        train_json = os.path.join(bdd_labels_dir, "bdd100k_labels_images_train.json")
        val_json = os.path.join(bdd_labels_dir, "bdd100k_labels_images_val.json")

        if os.path.isdir(bdd_images_train) and os.path.isfile(train_json):
            train_dataset = BDDTimeOfDayDataset(bdd_images_train, train_json, transform=transform)
            val_dataset = BDDTimeOfDayDataset(bdd_images_val, val_json, transform=transform)
        else:
            raise RuntimeError(f"No suitable training data found. Expected either {train_folder} or BDD data under {os.path.join(DATA_DIR, 'raw')}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ------------------------------
    # MODEL
    # ------------------------------
    encoder = get_resnet()  # ResNetSimCLR backbone

    # ------------------------------
    # LOAD SIMCLR CHECKPOINT
    # ------------------------------
    if not os.path.isfile(SIMCLR_CKPT):
        raise RuntimeError(f"SimCLR checkpoint not found at {SIMCLR_CKPT}")

    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)
    state_dict = ckpt.get("model_state", ckpt)

    # Rename encoder.* -> backbone.* and keep only keys present in the encoder
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_key = "backbone." + k[len("encoder."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Filter to keys that exist in encoder and load non-strictly (ignore projector keys)
    enc_keys = set(encoder.state_dict().keys())
    filtered = {k: v for k, v in new_state_dict.items() if k in enc_keys}
    missing_or_unexpected = set(new_state_dict.keys()) - set(filtered.keys())
    if missing_or_unexpected:
        print(f"Ignoring checkpoint keys not used by encoder: {sorted(list(missing_or_unexpected))[:5]}{'...' if len(missing_or_unexpected)>5 else ''}")

    encoder.load_state_dict(filtered, strict=False)

    encoder = encoder.to(DEVICE)
    encoder.eval()  # Freeze encoder

    # ResNetSimCLR backbone outputs 2048-d features
    classifier = nn.Linear(2048, NUM_CLASSES).to(DEVICE)

    # ------------------------------
    # LOSS & OPTIMIZER
    # ------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LR)

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Forward pass through frozen encoder
            with torch.no_grad():
                features = encoder(imgs)  # ResNetSimCLR returns 2048-d features

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

    # ------------------------------
    # SAVE MODEL
    # ------------------------------
    os.makedirs("lane_segmentation", exist_ok=True)
    torch.save(classifier.state_dict(), "lane_segmentation/linear_classifier.pth")
    print("Linear classifier saved to lane_segmentation/linear_classifier.pth")


if __name__ == '__main__':
    # On Windows, protect the entry point for multiprocessing
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
