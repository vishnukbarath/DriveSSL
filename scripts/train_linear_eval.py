import os
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ssl_dataset import SSLImageDataset
from src.models.simclr import SimCLR
from src.losses.nt_xent_custom import nt_xent_custom as nt_xent_loss
from src.utils.device import get_device


# ---------------- CONFIG ---------------- #

DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4
TEMPERATURE = 0.5
FEATURE_DIM = 128

SAVE_DIR = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------- #


def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    dataset = SSLImageDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    model = SimCLR(feature_dim=FEATURE_DIM).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch}/{EPOCHS}]")

        for x1, x2 in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"simclr_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "encoder": model.encoder.state_dict(),
                "projector": model.projector.state_dict(),
                "optimizer": optimizer.state_dict()
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
