import torch
import torch.nn as nn
from src.models.resnet_simclr import get_resnet


class MultiHeadSimCLR(nn.Module):
    """
    SimCLR encoder with multiple supervised heads
    """

    def __init__(self, num_time=3, num_weather=4, num_domain=2):
        super().__init__()

        self.encoder = get_resnet()  # 2048-D output

        self.time_head = nn.Linear(2048, num_time)
        self.weather_head = nn.Linear(2048, num_weather)
        self.domain_head = nn.Linear(2048, num_domain)

    def forward(self, x):
        feats = self.encoder(x)

        return {
            "time": self.time_head(feats),
            "weather": self.weather_head(feats),
            "domain": self.domain_head(feats),
            "features": feats
        }
