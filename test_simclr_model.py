import torch
from src.models.simclr import SimCLR

def main():
    model = SimCLR(feature_dim=128)

    x = torch.randn(8, 3, 224, 224)
    z = model(x)

    print("Output shape:", z.shape)
    print("Embedding norm (should be ~1):", torch.norm(z, dim=1))

if __name__ == "__main__":
    main()
