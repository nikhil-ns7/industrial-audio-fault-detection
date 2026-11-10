import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from audio_dataset import MIMIIDataset


TRAIN_CSV = r"C:\Fault_Recognition\data\splits\train.csv"


print("=" * 60)
print("DATASET LOADER TEST")
print("=" * 60)


dataset = MIMIIDataset(TRAIN_CSV)

print("\nDataset size:")
print(len(dataset))


# Test first recording
waveform, label = dataset[0]

print("\nFirst recording:")
print("Waveform shape:", waveform.shape)
print("Label:", label.item())


# Test a few more recordings
for index in [1, 10, 100]:

    waveform, label = dataset[index]

    print(
        f"Index {index}: "
        f"shape={waveform.shape}, "
        f"label={label.item()}"
    )


print("\n" + "=" * 60)
print("DATASET LOADER TEST COMPLETE")
print("=" * 60)