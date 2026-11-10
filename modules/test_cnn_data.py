import torch
from torch.utils.data import DataLoader
from cnn_dataset import CNNMIMIIDataset
from cnn_model import CNNFaultDetector

TRAIN_CSV = r"C:\Fault_Recognition\data\splits\train.csv"

print("Loading dataset...")

dataset = CNNMIMIIDataset(TRAIN_CSV)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print("Loading one batch...")

mel, labels = next(iter(loader))

print("Mel batch shape:", mel.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)

model = CNNFaultDetector()

output = model(mel)

print("CNN output shape:", output.shape)
print("CNN test successful!")