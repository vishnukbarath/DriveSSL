import torch
from src.losses.nt_xent import NTXentLoss

def main():
    batch_size = 8
    feature_dim = 128

    z1 = torch.randn(batch_size, feature_dim)
    z2 = torch.randn(batch_size, feature_dim)

    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    loss_fn = NTXentLoss(temperature=0.5)
    loss = loss_fn(z1, z2)

    print("NT-Xent loss value:", loss.item())

if __name__ == "__main__":
    main()
