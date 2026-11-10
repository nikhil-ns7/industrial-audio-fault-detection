import os
import csv
import zipfile
import io

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from transformers import AutoFeatureExtractor

from ast_model import ASTFaultDetector
from audio_preprocessing import preprocess_waveform


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
    "ast_best.pth"
)

RAW_DIR = os.path.join(
    ROOT,
    "data",
    "raw"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

BATCH_SIZE = 4

LABEL_MAP = {
    "normal": 0,
    "abnormal": 1
}


# ============================================================
# SAFE AST DATASET
# ============================================================

class SafeASTMIMIIDataset(Dataset):

    def __init__(self, csv_path):

        self.rows = []

        # Use built-in csv instead of pandas
        with open(
            csv_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                self.rows.append(row)

        print(
            f"Loaded {len(self.rows)} test samples."
        )

    def __len__(self):
        return len(self.rows)

    def load_audio(self, row):

        source = row["source"]
        file_path = row["file_path"]

        # ----------------------------------------------------
        # Pump dataset
        # ----------------------------------------------------

        if source == "0_dB_pump":

            audio_path = file_path

            audio, sample_rate = sf.read(
                audio_path
            )

            waveform = torch.from_numpy(
                audio
            ).float()

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

        return (
            waveform,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# LOAD FEATURE EXTRACTOR
# ============================================================

print("=" * 60)
print("AST SAFE EVALUATION")
print("=" * 60)

print("\nLoading AST feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Feature extractor loaded successfully.")


# ============================================================
# DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = SafeASTMIMIIDataset(
    TEST_CSV
)


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    waveforms = []
    labels = []

    for waveform, label in batch:

        waveforms.append(
            waveform.numpy()
        )

        labels.append(
            label.item()
        )

    inputs = feature_extractor(
        waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return (
        inputs["input_values"],
        labels
    )


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", DEVICE)


# ============================================================
# LOAD AST MODEL
# ============================================================

print("\nLoading AST model...")

model = ASTFaultDetector()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)

model.eval()

print("AST model loaded successfully.")


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
        input_values,
        labels
    ) in enumerate(test_loader):

        input_values = input_values.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        outputs = model(
            input_values
        )

        predictions = torch.argmax(
            outputs,
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
        ) % 20 == 0:

            print(
                f"Processed "
                f"{total}/{len(test_dataset)} samples"
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

# Abnormal = positive class

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


# Normal class

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
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL AST TEST RESULTS")
print("=" * 60)

print(
    f"\nTest samples : {total}"
)

print(
    f"Correct      : {correct}"
)

print(
    f"Incorrect    : {total - correct}"
)

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
print("AST EVALUATION COMPLETE")
print("=" * 60)