import torch
import time
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.models.multihead_model import MultiHeadSimCLR
from src.utils.device import get_device


def evaluate(model, loader, device):
    model.eval()
    correct = {"time": 0, "weather": 0, "domain": 0}
    total = 0

    with torch.no_grad():
        for imgs, t, w, d in loader:
            imgs = imgs.to(device)
            t, w, d = t.to(device), w.to(device), d.to(device)

            out = model(imgs)
            correct["time"] += (out["time"].argmax(1) == t).sum().item()
            correct["weather"] += (out["weather"].argmax(1) == w).sum().item()
            correct["domain"] += (out["domain"].argmax(1) == d).sum().item()
            total += imgs.size(0)

    return {k: 100 * v / total for k, v in correct.items()}


def main():
    device = get_device()
    print("[INFO] Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BDDMultiHeadDataset(
        img_dir="data/bdd100k/images/100k/train",
        label_json="data/bdd100k/labels/bdd100k_labels_images_train.json",
        transform=transform,
    )

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    model = MultiHeadSimCLR().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val = 0

    for epoch in range(1, 21):
        model.train()
        start = time.time()
        total_loss = 0

        for imgs, t, w, d in train_loader:
            imgs = imgs.to(device)
            t, w, d = t.to(device), w.to(device), d.to(device)

            out = model(imgs)
            loss = (
                loss_fn(out["time"], t)
                + loss_fn(out["weather"], w)
                + loss_fn(out["domain"], d)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        avg_val = sum(val_acc.values()) / 3

        print(
            f"Epoch {epoch} | "
            f"Loss {total_loss:.2f} | "
            f"Val Time {val_acc['time']:.2f}% | "
            f"Val Weather {val_acc['weather']:.2f}% | "
            f"Val Domain {val_acc['domain']:.2f}% | "
            f"AVG {avg_val:.2f}% | "
            f"{time.time()-start:.1f}s"
        )

        if avg_val > best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "experiments/multihead/best_val_model.pth")
            print("[INFO] Saved best validation model")

    print("[DONE] Best Validation Accuracy:", best_val)


if __name__ == "__main__":
    main()
