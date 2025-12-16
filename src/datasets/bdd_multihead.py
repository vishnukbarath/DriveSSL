import json
import os
from PIL import Image
from torch.utils.data import Dataset

TIME_MAP = {
    "daytime": 0,
    "night": 1,
    "dawn/dusk": 2,
}

WEATHER_MAP = {
    "clear": 0,
    "partly cloudy": 1,
    "overcast": 2,
    "rainy": 3,
    "snowy": 4,
    "foggy": 5,
}

DOMAIN_MAP = {
    "city street": 0,
    "highway": 1,
}


class BDDMultiHeadDataset(Dataset):
    def __init__(self, img_dir, label_json, transform=None):
        self.samples = []
        self.transform = transform

        with open(label_json, "r") as f:
            data = json.load(f)

        for item in data:
            img_path = os.path.join(img_dir, item["name"])
            if not os.path.exists(img_path):
                continue

            attrs = item.get("attributes", {})

            time = TIME_MAP.get(attrs.get("timeofday"))
            weather = WEATHER_MAP.get(attrs.get("weather"))
            domain = DOMAIN_MAP.get(attrs.get("scene"))

            # STRICT REQUIREMENT — all heads present
            if time is None or weather is None or domain is None:
                continue

            self.samples.append((img_path, time, weather, domain))

        print(f"[INFO] Multi-head samples loaded: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("ZERO samples loaded — dataset paths or labels are wrong.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, t, w, d = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, t, w, d
