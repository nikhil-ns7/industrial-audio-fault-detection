import os
import csv
import random
import zipfile

# Project paths
ROOT = r"C:\Fault_Recognition"
RAW = os.path.join(ROOT, "data", "raw")
OUTPUT = os.path.join(ROOT, "master_dataset.csv")

random.seed(42)

# Use the same number of normal and abnormal recordings
TARGETS = {
    "pump": 313,
    "fan": 1475,
    "slider": 890,
    "valve": 479
}

rows = []


def collect_from_pump():
    pump_root = os.path.join(RAW, "0_dB_pump", "pump")

    print("\nReading extracted pump dataset...")

    for label in ["normal", "abnormal"]:
        files = []

        for current_root, dirs, filenames in os.walk(pump_root):
            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    path = os.path.join(current_root, filename)

                    if os.sep + label + os.sep in path.lower():
                        files.append(path)

        target = TARGETS["pump"]

        if len(files) < target:
            raise ValueError(
                f"Pump {label}: only {len(files)} available, "
                f"but {target} required."
            )

        selected = random.sample(files, target)

        for path in selected:
            rows.append({
                "machine": "pump",
                "label": label,
                "source": "0_dB_pump",
                "file_path": path
            })

        print(f"  {label}: {len(selected)} selected")


def collect_from_zip(zip_path, machine):
    print(f"\nReading {os.path.basename(zip_path)}...")

    with zipfile.ZipFile(zip_path, "r") as z:
        entries = [
            e.filename
            for e in z.infolist()
            if e.filename.lower().endswith(".wav")
        ]

    for label in ["normal", "abnormal"]:

        files = [
            e for e in entries
            if f"/{label}/" in e.replace("\\", "/").lower()
        ]

        target = TARGETS[machine]

        if len(files) < target:
            raise ValueError(
                f"{machine} {label}: only {len(files)} available, "
                f"but {target} required."
            )

        selected = random.sample(files, target)

        for path in selected:
            rows.append({
                "machine": machine,
                "label": label,
                "source": os.path.basename(zip_path),
                "file_path": path
            })

        print(f"  {label}: {len(selected)} selected")


# -----------------------------
# Collect all four machines
# -----------------------------

collect_from_pump()

for machine in ["fan", "slider", "valve"]:

    zip_path = os.path.join(
        RAW,
        f"0_dB_{machine}.zip"
    )

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Dataset not found: {zip_path}"
        )

    collect_from_zip(zip_path, machine)


# Shuffle dataset
random.shuffle(rows)


# Save master CSV
with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "machine",
            "label",
            "source",
            "file_path"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# -----------------------------
# Verification
# -----------------------------

normal = sum(
    r["label"] == "normal"
    for r in rows
)

abnormal = sum(
    r["label"] == "abnormal"
    for r in rows
)

print("\n" + "=" * 55)
print("MASTER DATASET CREATED")
print("=" * 55)

print(f"Total recordings : {len(rows)}")
print(f"Normal           : {normal}")
print(f"Abnormal         : {abnormal}")

print("\nMachine distribution:")

for machine in TARGETS:
    count = sum(
        r["machine"] == machine
        for r in rows
    )
    print(f"  {machine:<8}: {count}")

print("\nSaved to:")
print(OUTPUT)

print("=" * 55)