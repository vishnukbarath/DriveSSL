import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SSLImageDataset(Dataset):
    def __init__(self, image_dir, image_size=224):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.image_paths) == 0:
            raise RuntimeError("No images found in SSL dataset directory.")

        # SimCLR-style augmentations
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Two different augmented views of the same image
        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2
