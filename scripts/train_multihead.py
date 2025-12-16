import json
import os
from PIL import Image
from torch.utils.data import Dataset

# ---- LABEL MAPS (COMPLETE) ---- #

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
        self.img_dir = img_dir
        self.transform = transform

        with open(label_json, "r") as f:
            raw_data = json.load(f)

        self.samples = []

        for item in raw_data:
            img_name = item["name"]
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            attrs = item.get("attributes", {})

            # Skip samples with missing labels
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

        print(f"[INFO] Loaded {len(self.samples)} multi-head samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, time_y, weather_y, domain_y = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "time": time_y,
            "weather": weather_y,
            "domain": domain_y,
        }
