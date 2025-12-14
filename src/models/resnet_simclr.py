import torch
import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):
    """
    ResNet-50 backbone used for SimCLR and downstream tasks.
    Output feature dimension: 2048
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        backbone.fc = nn.Identity()   # remove classifier
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def get_resnet():
    """
    Factory function for downstream usage (linear eval, finetuning, etc.)
    """
    return ResNetSimCLR()
