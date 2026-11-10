import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from cnn_dataset import CNNMIMIIDataset


CSV = r"C:\Fault_Recognition\data\splits\train.csv"

dataset = CNNMIMIIDataset(CSV)

print("=" * 60)
print("CNN DATASET TEST")
print("=" * 60)

print("Dataset size:", len(dataset))

mel, label = dataset[0]

print("Mel spectrogram shape:", mel.shape)
print("Label:", label.item())

print("=" * 60)
print("CNN DATASET TEST COMPLETE")
print("=" * 60)