import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

WEATHER_MAP = {
    "clear": 0,
    "rainy": 1,
    "snowy": 2,
    "foggy": 3
}

SCENE_MAP = {
    "city street": 0,
    "highway": 1,
    "residential": 2
}

TIME_MAP = {
    "daytime": 0,
    "night": 1,
    "dawn/dusk": 2
}

class BDDMultiHeadDataset(Dataset):
    def __init__(self, image_root, label_json):
        self.image_root = image_root

        with open(label_json, "r") as f:
            raw_labels = json.load(f)

        # filename → attributes
        self.label_map = {}
        for item in raw_labels:
            attrs = item.get("attributes", {})
            if (
                attrs.get("weather") in WEATHER_MAP and
                attrs.get("scene") in SCENE_MAP and
                attrs.get("timeofday") in TIME_MAP
            ):
                self.label_map[item["name"]] = attrs

        self.samples = []
        for root, _, files in os.walk(image_root):
            for file in files:
                if file.endswith(".jpg") and file in self.label_map:
                    self.samples.append(os.path.join(root, file))

        print(f"[INFO] Loaded {len(self.samples)} valid samples")

        if len(self.samples) == 0:
            raise RuntimeError("NO VALID SAMPLES FOUND — CHECK DATASET PATHS")

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        attrs = self.label_map[os.path.basename(img_path)]

        targets = {
            "weather": WEATHER_MAP[attrs["weather"]],
            "scene": SCENE_MAP[attrs["scene"]],
            "timeofday": TIME_MAP[attrs["timeofday"]],
        }

        return self.transform(img), targets
