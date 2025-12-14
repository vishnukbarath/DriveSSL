import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.datasets.ssl_dataset import SSLImageDataset
from src.models.simclr import SimCLR
from src.losses.nt_xent import NTXentLoss

# ================= CONFIG =================
DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data\ssl_unlabeled"
BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-4
FEATURE_DIM = 128
TEMPERATURE = 0.5
NUM_WORKERS = 4
CHECKPOINT_DIR = "experiments/simclr/checkpoints"

# ==========================================

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = SSLImageDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    model = SimCLR(feature_dim=FEATURE_DIM).to(device)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scaler = GradScaler()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch}/{EPOCHS}]")

        for x1, x2 in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                z1 = model(x1)
                z2 = model(x2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"simclr_epoch_{epoch}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, ckpt_path)

if __name__ == "__main__":
    main()
