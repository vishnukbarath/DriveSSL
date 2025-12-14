import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TIME_MAP = {
    "daytime": 0,
    "night": 1,
    "dawn/dusk": 2
}

class BDDTimeOfDayDataset(Dataset):
    def __init__(self, images_dir, label_json, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(label_json, "r") as f:
            data = json.load(f)

        missing = 0
        for item in data:
            img_name = item["name"]
            tod = item["attributes"]["timeofday"]
            img_path = os.path.join(self.images_dir, img_name)
            if tod in TIME_MAP:
                if not os.path.isfile(img_path):
                    missing += 1
                    continue
                self.samples.append((img_name, TIME_MAP[tod]))

        if missing:
            print(f"Warning: {missing} samples referenced in labels were not found under {images_dir} and were skipped.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
