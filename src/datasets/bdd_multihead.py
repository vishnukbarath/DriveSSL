import json
import os
from PIL import Image
from torch.utils.data import Dataset

TIME_MAP = {
    "daytime": 0,
    "night": 1,
    "dawn/dusk": 2
}

WEATHER_MAP = {
    "clear": 0,
    "partly cloudy": 1,
    "overcast": 2,
    "rainy": 3,
    "snowy": 4,
    "foggy": 5
}

DOMAIN_MAP = {
    "city street": 0,
    "highway": 1
}


class BDDMultiHeadDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None):
        self.samples = []
        self.transform = transform

        with open(label_json, "r") as f:
            data = json.load(f)

        for item in data:
            attrs = item.get("attributes", {})
            img_path = os.path.join(img_dir, item["name"])

            if not os.path.exists(img_path):
                continue

            if (
                attrs.get("timeofday") not in TIME_MAP
                or attrs.get("weather") not in WEATHER_MAP
                or attrs.get("scene") not in DOMAIN_MAP
            ):
                continue

            self.samples.append(
                (
                    img_path,
                    TIME_MAP[attrs["timeofday"]],
                    WEATHER_MAP[attrs["weather"]],
                    DOMAIN_MAP[attrs["scene"]],
                )
            )

        print(f"[INFO] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, t, w, d = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, t, w, d
