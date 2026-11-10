import os
import csv
import zipfile
import io

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from cnn_model import CNNFaultDetector
from audio_preprocessing import preprocess_waveform, waveform_to_mel


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Fault_Recognition"

TEST_CSV = os.path.join(
    ROOT,
    "data",
    "splits",
    "test.csv"
)

MODEL_PATH = os.path.join(
    ROOT,
    "model",
    "cnn_best.pth"
)

RAW_DIR = os.path.join(
    ROOT,
    "data",
    "raw"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16

LABEL_MAP = {
    "normal": 0,
    "abnormal": 1
}


# ============================================================
# SAFE CSV DATASET
# ============================================================

class SafeCNNMIMIIDataset(Dataset):

    def __init__(self, csv_path):

        self.rows = []

        with open(
            csv_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                self.rows.append(row)

        print(f"Loaded {len(self.rows)} test samples.")

    def __len__(self):
        return len(self.rows)

    def load_audio(self, row):

        source = row["source"]
        file_path = row["file_path"]

        # ----------------------------------------------------
        # Extracted pump dataset
        # ----------------------------------------------------

        if source == "0_dB_pump":

            audio_path = file_path

            audio, sample_rate = sf.read(
                audio_path
            )

            waveform = torch.from_numpy(
                audio
            ).float()

            # Same orientation as original audio_dataset.py
            if waveform.ndim == 2:
                waveform = waveform.T

            waveform = preprocess_waveform(
                waveform,
                sample_rate
            )

            return waveform

        # ----------------------------------------------------
        # ZIP datasets
        # ----------------------------------------------------

        else:

            zip_path = os.path.join(
                RAW_DIR,
                source
            )

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as z:

                with z.open(file_path) as audio_file:

                    audio_bytes = audio_file.read()

            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32"
            )

            waveform = torch.from_numpy(
                audio
            ).float()

            # Same orientation as original audio_dataset.py
            if waveform.ndim == 2:
                waveform = waveform.T

            waveform = preprocess_waveform(
                waveform,
                sample_rate
            )

            return waveform

    def __getitem__(self, index):

        row = self.rows[index]

        waveform = self.load_audio(row)

        label = LABEL_MAP[
            row["label"]
        ]

        # Same Mel pipeline as cnn_dataset.py
        mel = waveform_to_mel(
            waveform,
            sample_rate=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )

        # Add CNN channel dimension
        # [128, time] -> [1, 128, time]
        mel = mel.unsqueeze(0)

        return (
            mel,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CNN SAFE EVALUATION")
print("=" * 60)

print("\nTest CSV:")
print(TEST_CSV)

print("\nModel:")
print(MODEL_PATH)

dataset = SafeCNNMIMIIDataset(
    TEST_CSV
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

model = CNNFaultDetector()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Handle common checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Could already be a raw state_dict
        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.to(device)

model.eval()

print("CNN model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []

correct = 0
total = 0


print("\nRunning evaluation...\n")


with torch.no_grad():

    for batch_index, (
        mel,
        labels
    ) in enumerate(loader):

        mel = mel.to(device)
        labels = labels.to(device)

        outputs = model(mel)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        if (
            batch_index + 1
        ) % 10 == 0:

            print(
                f"Processed "
                f"{total}/{len(dataset)} samples"
            )


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn = 0
fp = 0
fn = 0
tp = 0

for actual, predicted in zip(
    all_labels,
    all_predictions
):

    if actual == 0 and predicted == 0:
        tn += 1

    elif actual == 0 and predicted == 1:
        fp += 1

    elif actual == 1 and predicted == 0:
        fn += 1

    elif actual == 1 and predicted == 1:
        tp += 1


# ============================================================
# METRICS
# ============================================================

accuracy = (
    (tp + tn) / total
    if total > 0
    else 0
)

precision_abnormal = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall_abnormal = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1_abnormal = (
    2
    * precision_abnormal
    * recall_abnormal
    / (
        precision_abnormal
        + recall_abnormal
    )
    if (
        precision_abnormal
        + recall_abnormal
    ) > 0
    else 0
)


# Normal-class metrics

precision_normal = (
    tn / (tn + fn)
    if (tn + fn) > 0
    else 0
)

recall_normal = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1_normal = (
    2
    * precision_normal
    * recall_normal
    / (
        precision_normal
        + recall_normal
    )
    if (
        precision_normal
        + recall_normal
    ) > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL CNN TEST RESULTS")
print("=" * 60)

print(f"\nTest samples : {total}")
print(f"Correct      : {correct}")
print(f"Incorrect    : {total - correct}")

print(
    f"\nAccuracy     : "
    f"{accuracy * 100:.2f}%"
)

print("\n--- Normal ---")

print(
    f"Precision    : "
    f"{precision_normal * 100:.2f}%"
)

print(
    f"Recall       : "
    f"{recall_normal * 100:.2f}%"
)

print(
    f"F1-score     : "
    f"{f1_normal * 100:.2f}%"
)

print("\n--- Abnormal ---")

print(
    f"Precision    : "
    f"{precision_abnormal * 100:.2f}%"
)

print(
    f"Recall       : "
    f"{recall_abnormal * 100:.2f}%"
)

print(
    f"F1-score     : "
    f"{f1_abnormal * 100:.2f}%"
)

print("\n--- Confusion Matrix ---")

print(
    "                 Predicted"
)

print(
    "                 Normal  Abnormal"
)

print(
    f"Actual Normal    {tn:6d}  {fp:8d}"
)

print(
    f"Actual Abnormal  {fn:6d}  {tp:8d}"
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)