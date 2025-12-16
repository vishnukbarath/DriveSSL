import torch
import torch.nn as nn
from src.models.resnet_simclr import get_resnet


class MultiHeadSimCLR(nn.Module):
    def __init__(self, num_time=3, num_weather=6, num_domain=2):
        super().__init__()

        self.encoder = get_resnet()  # 2048-d output

        self.head_time = nn.Linear(2048, num_time)
        self.head_weather = nn.Linear(2048, num_weather)
        self.head_domain = nn.Linear(2048, num_domain)

    def forward(self, x):
        feats = self.encoder(x)

        return {
            "time": self.head_time(feats),
            "weather": self.head_weather(feats),
            "domain": self.head_domain(feats),
        }
