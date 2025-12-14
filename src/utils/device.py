import torch

def get_device():
    """
    Returns the appropriate device (cuda if available, else cpu).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
