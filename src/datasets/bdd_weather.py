import os
import json
from PIL import Image
from torch.utils.data import Dataset

WEATHER_MAP = {
    "clear": 0,
    "rainy": 1,
    "snowy": 2,
    "foggy": 3
}

class BDDWeatherDataset(Dataset):
    def __init__(self, image_dir, label_json, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(label_json, "r") as f:
            labels = json.load(f)

        missing = 0
        for item in labels:
            weather = item["attributes"].get("weather", None)
            if weather not in WEATHER_MAP:
                continue

            img_path = os.path.join(image_dir, item["name"])
            if not os.path.isfile(img_path):
                missing += 1
                continue

            self.samples.append((img_path, WEATHER_MAP[weather]))

        if missing > 0:
            print(f"[WARN] {missing} images missing and skipped")

        print(f"[INFO] Loaded {len(self.samples)} weather samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
