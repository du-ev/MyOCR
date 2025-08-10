import torch
from crnn import CRNN

# I was trying to find how the cnn actually changes the shape of an input vector
model = CRNN(num_classes=10)
model.eval()

# (batch, channels, h, w)
dummy_input = torch.randn(1, 3, 64, 360)

with torch.no_grad():
    cnn_output = model.cnn(dummy_input)

print(f"Input shape:  {dummy_input.shape}")
print(f"CNN Output shape: {cnn_output.shape}")
