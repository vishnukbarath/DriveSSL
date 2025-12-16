import torch

state = torch.load(
    r"C:\Users\vishn\Documents\DriveSSL\experiments\weather\weather_linear.pth"
)
print(state.keys())
