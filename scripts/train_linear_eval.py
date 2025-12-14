import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------------- Local imports ---------------- #
from src.datasets.bdd_linear import BDDTimeOfDayDataset
from src.models.resnet_simclr import get_resnet

# ---------------- CONFIG ---------------- #
BATCH_SIZE = 256
EPOCHS = 20
LR = 0.001
NUM_CLASSES = 3  # day / night / dawn-dusk

IMAGES_DIR = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k\images\100k\train"
LABEL_JSON = r"C:\Users\vishn\Documents\DriveSSL\data\raw\bdd100k_original\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
SIMCLR_CKPT = r"C:\Users\vishn\Documents\DriveSSL\experiments\simclr\checkpoints\simclr_epoch_20.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMS ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- MAIN ---------------- #
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # Dataset and loader
    dataset = BDDTimeOfDayDataset(images_dir=IMAGES_DIR, label_json=LABEL_JSON, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # Load encoder
    encoder = get_resnet().to(DEVICE)
    ckpt = torch.load(SIMCLR_CKPT, map_location=DEVICE)

    # Load only model weights
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()  # freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Linear classifier
    classifier = nn.Linear(2048, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LR)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(loader, desc=f"Epoch [{epoch}/{EPOCHS}]")
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                features = encoder(images)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

        # Save classifier every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join("linear_checkpoints", f"classifier_epoch_{epoch}.pth")
            os.makedirs("linear_checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "classifier_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, save_path)
            print(f"[INFO] Saved classifier checkpoint at {save_path}")

if __name__ == "__main__":
    main()
