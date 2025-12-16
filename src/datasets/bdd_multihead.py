import os
import json
from PIL import Image
from torch.utils.data import Dataset


TIME_MAP = {"daytime": 0, "night": 1, "dawn/dusk": 2}
WEATHER_MAP = {"clear": 0, "rainy": 1, "foggy": 2, "snowy": 3}
DOMAIN_MAP = {"daytime": 0, "night": 1}


class BDDMultiHeadDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(label_json, "r") as f:
            raw = json.load(f)

        self.samples = []
        for item in raw:
            img_path = os.path.join(img_dir, item["name"])
            if not os.path.isfile(img_path):
                continue

            attrs = item["attributes"]
            self.samples.append((
                img_path,
                TIME_MAP[attrs["timeofday"]],
                WEATHER_MAP[attrs["weather"]],
                DOMAIN_MAP["daytime" if attrs["timeofday"] == "daytime" else "night"]
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t, w, d = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, t, w, d
