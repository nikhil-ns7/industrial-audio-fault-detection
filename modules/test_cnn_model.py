import torch
from cnn_model import CNNFaultDetector

model = CNNFaultDetector()

x = torch.randn(4, 1, 128, 313)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)
print("Output:")
print(output)