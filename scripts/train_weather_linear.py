import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet_simclr import get_resnet
from src.datasets.bdd_weather import BDDWeatherDataset
from src.utils.device import get_device

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
NUM_CLASSES = 4

DEVICE = get_device()

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_train = os.path.join(DATA_DIR, "raw", "bdd100k_original", "bdd100k",
                             "bdd100k", "images", "100k", "train")
    img_val = os.path.join(DATA_DIR, "raw", "bdd100k_original", "bdd100k",
                           "bdd100k", "images", "100k", "val")

    labels = os.path.join(DATA_DIR, "raw", "bdd100k_original",
                          "bdd100k_labels_release", "bdd100k", "labels")

    train_json = os.path.join(labels, "bdd100k_labels_images_train.json")
    val_json = os.path.join(labels, "bdd100k_labels_images_val.json")

    train_ds = BDDWeatherDataset(img_train, train_json, transform)
    val_ds = BDDWeatherDataset(img_val, val_json, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # -------- Encoder --------
    encoder = get_resnet().to(DEVICE)
    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)

    filtered = {k.replace("encoder.", "backbone."): v
                for k, v in state.items()
                if k.startswith("encoder.")}

    encoder.load_state_dict(filtered, strict=False)
    encoder.eval()

    # -------- Classifier --------
    classifier = nn.Linear(2048, NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LR)

    # -------- Training --------
    best_acc = 0.0
    start = time.time()

    for epoch in range(EPOCHS):
        classifier.train()
        correct, total, loss_sum = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                feats = encoder(x)

            out = classifier(feats)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        # -------- Validation --------
        classifier.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = classifier(encoder(x)).argmax(1)
                v_correct += preds.eq(y).sum().item()
                v_total += y.size(0)

        val_acc = 100 * v_correct / v_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("experiments/weather", exist_ok=True)
            torch.save(classifier.state_dict(),
                       "experiments/weather/weather_linear.pth")
            print("[INFO] Saved best model")

    print(f"[DONE] Best Val Acc: {best_acc:.2f}%")
    print(f"Total time: {(time.time()-start)/60:.2f} min")


if __name__ == "__main__":
    main()
