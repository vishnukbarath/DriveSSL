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


DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4

DEVICE = get_device()


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_img = os.path.join(DATA_DIR, "raw", "bdd100k_original", "bdd100k",
                             "bdd100k", "images", "100k", "train")
    train_json = os.path.join(DATA_DIR, "raw", "bdd100k_original",
                              "bdd100k_labels_release", "bdd100k", "labels",
                              "bdd100k_labels_images_train.json")

    dataset = BDDMultiHeadDataset(train_img, train_json, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = MultiHeadSimCLR().to(DEVICE)

    # Load SimCLR encoder weights
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)

    encoder_dict = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            encoder_dict["encoder.backbone." + k[len("encoder."):]] = v

    model.load_state_dict(encoder_dict, strict=False)
    print("[INFO] Loaded SimCLR encoder")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_loss = 0.0
        for imgs, t, w, d in loader:
            imgs, t, w, d = imgs.to(DEVICE), t.to(DEVICE), w.to(DEVICE), d.to(DEVICE)

            out = model(imgs)

            loss = (
                ce(out["time"], t) +
                ce(out["weather"], w) +
                ce(out["domain"], d)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {total_loss/len(loader):.4f} "
              f"Time: {time.time()-start:.2f}s")

    os.makedirs("experiments/multihead", exist_ok=True)
    torch.save(model.state_dict(), "experiments/multihead/multihead_model.pth")
    print("[DONE] Multi-head model saved")


if __name__ == "__main__":
    main()
