# src/datasets/ssl_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SSLImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # Ensure all files ending with .jpg, .jpeg, .png are included
        self.images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in SSL dataset directory: {root_dir}")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        # For SimCLR, return two different augmentations of the same image
        return self.transform(image), self.transform(image)
