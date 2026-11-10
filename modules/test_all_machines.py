import os
import sys
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from audio_dataset import MIMIIDataset


CSV = r"C:\Fault_Recognition\master_dataset.csv"

df = pd.read_csv(CSV)

print("=" * 60)
print("TESTING ALL FOUR MACHINES")
print("=" * 60)

for machine in ["pump", "fan", "slider", "valve"]:

    machine_df = df[df["machine"] == machine]

    # Save temporary CSV for this machine
    temp_csv = os.path.join(
        os.path.dirname(CSV),
        f"_test_{machine}.csv"
    )

    machine_df.to_csv(
        temp_csv,
        index=False
    )

    dataset = MIMIIDataset(temp_csv)

    waveform, label = dataset[0]

    print(
        f"{machine:8} → "
        f"records={len(dataset):4} | "
        f"shape={tuple(waveform.shape)} | "
        f"label={label.item()}"
    )

    os.remove(temp_csv)


print("=" * 60)
print("ALL MACHINE TEST COMPLETE")
print("=" * 60)