import torch.nn as nn
from src.models.resnet_simclr import get_resnet


class MultiHeadSimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_resnet()

        self.time_head = nn.Linear(2048, 3)
        self.weather_head = nn.Linear(2048, 6)
        self.domain_head = nn.Linear(2048, 2)

    def forward(self, x):
        feat = self.encoder(x)
        return {
            "time": self.time_head(feat),
            "weather": self.weather_head(feat),
            "domain": self.domain_head(feat),
        }
