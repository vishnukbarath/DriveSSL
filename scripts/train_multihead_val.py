import torch
from torch.utils.data import DataLoader
from src.datasets.bdd_multihead import BDDMultiHeadDataset
from src.models.multihead_model import MultiHeadSimCLR

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    dataset = BDDMultiHeadDataset(
        image_root=r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\bdd100k\images\100k\train",
        label_json=r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = MultiHeadSimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for imgs, targets in loader:
            imgs = imgs.to(device)

            out = model(imgs)

            loss = (
                loss_fn(out["weather"], targets["weather"].to(device)) +
                loss_fn(out["scene"], targets["scene"].to(device)) +
                loss_fn(out["timeofday"], targets["timeofday"].to(device))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "bdd_multihead.pth")
    print("[INFO] Model saved: bdd_multihead.pth")

if __name__ == "__main__":
    main()
