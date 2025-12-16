import json
import os
from PIL import Image
from torch.utils.data import Dataset

# ---------------- LABEL MAPS ---------------- #

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
    """
    Multi-head dataset for BDD100K
    Outputs:
      - time of day
      - weather
      - scene domain
    """

    def __init__(self, img_dir, label_json, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(label_json, "r") as f:
            raw_data = json.load(f)

        self.samples = []
        skipped = 0

        for item in raw_data:
            img_name = item.get("name")
            if img_name is None:
                skipped += 1
                continue

            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            attrs = item.get("attributes", {})

            time_key = attrs.get("timeofday")
            weather_key = attrs.get("weather")
            domain_key = attrs.get("scene")

            if (
                time_key not in TIME_MAP
                or weather_key not in WEATHER_MAP
                or domain_key not in DOMAIN_MAP
            ):
                skipped += 1
                continue

            self.samples.append(
                (
                    img_path,
                    TIME_MAP[time_key],
                    WEATHER_MAP[weather_key],
                    DOMAIN_MAP[domain_key],
                )
            )

        print(
            f"[INFO] Loaded {len(self.samples)} multi-head samples "
            f"(skipped {skipped})"
        )

        if len(self.samples) == 0:
            raise RuntimeError("No valid multi-head samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, time_y, weather_y, domain_y = self.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "time": time_y,
            "weather": weather_y,
            "domain": domain_y,
        }
