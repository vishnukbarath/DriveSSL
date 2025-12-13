from torch.utils.data import DataLoader
from src.datasets.ssl_dataset import SSLImageDataset

def main():
    DATA_DIR = r"C:\Users\vishn\Documents\DriveSSL\data\ssl_unlabeled"

    dataset = SSLImageDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # IMPORTANT for Windows testing
    )

    v1, v2 = next(iter(loader))

    print("View1 shape:", v1.shape)
    print("View2 shape:", v2.shape)

if __name__ == "__main__":
    main()
