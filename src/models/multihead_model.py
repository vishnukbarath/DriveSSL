import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadSimCLR(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = 512

        self.weather_head = nn.Linear(feat_dim, 4)
        self.scene_head = nn.Linear(feat_dim, 3)
        self.time_head = nn.Linear(feat_dim, 3)

    def forward(self, x):
        feats = self.encoder(x).squeeze(-1).squeeze(-1)

        return {
            "weather": self.weather_head(feats),
            "scene": self.scene_head(feats),
            "timeofday": self.time_head(feats)
        }
