import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet_simclr import ResNetSimCLR


class SimCLR(nn.Module):
    """
    Full SimCLR model = Encoder + Projection Head
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        self.encoder = ResNetSimCLR()

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)          # [B, 2048]
        z = self.projector(h)        # [B, feature_dim]
        z = F.normalize(z, dim=1)
        return z
