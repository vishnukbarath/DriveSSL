import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        # Backbone encoder (ResNet-50)
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()  # output: 2048-d

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)          # [B, 2048]
        z = self.projector(h)        # [B, feature_dim]
        z = F.normalize(z, dim=1)    # cosine similarity ready
        return z
