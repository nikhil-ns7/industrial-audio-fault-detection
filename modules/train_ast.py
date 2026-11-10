import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from transformers import AutoFeatureExtractor

from ast_dataset import ASTMIMIIDataset
from ast_model import ASTFaultDetector


# =========================
# PATHS
# =========================

ROOT = r"C:\Fault_Recognition"

TRAIN_CSV = os.path.join(
    ROOT, "data", "splits", "train.csv"
)

VAL_CSV = os.path.join(
    ROOT, "data", "splits", "validation.csv"
)

TEST_CSV = os.path.join(
    ROOT, "data", "splits", "test.csv"
)

MODEL_DIR = os.path.join(
    ROOT, "model"
)

RESULT_DIR = os.path.join(
    ROOT, "result"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# SETTINGS
# =========================

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("AST INDUSTRIAL FAULT DETECTION")
print("=" * 60)

print("Device:", DEVICE)


# =========================
# FEATURE EXTRACTOR
# =========================

print("\nLoading AST feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Feature extractor loaded.")


# =========================
# DATASET
# =========================

print("\nLoading datasets...")

train_dataset = ASTMIMIIDataset(TRAIN_CSV)
val_dataset = ASTMIMIIDataset(VAL_CSV)
test_dataset = ASTMIMIIDataset(TEST_CSV)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))


# =========================
# COLLATE FUNCTION
# =========================

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

    return inputs["input_values"], labels


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# =========================
# MODEL
# =========================

print("\nLoading AST model...")

model = ASTFaultDetector().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

print("AST model ready.")


# =========================
# EVALUATION
# =========================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0

    with torch.no_grad():

        for input_values, labels in loader:

            input_values = input_values.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                input_values
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# =========================
# TRAINING
# =========================

best_f1 = 0.0
best_epoch = 0

print("\nStarting AST training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_index, (
        input_values,
        labels
    ) in enumerate(train_loader):

        input_values = input_values.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            input_values
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/"
                f"{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = (
        total_loss /
        len(train_loader)
    )

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = evaluate(
        model,
        val_loader
    )

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy  : {val_accuracy:.4f}"
    )

    print(
        f"Val Precision : {val_precision:.4f}"
    )

    print(
        f"Val Recall    : {val_recall:.4f}"
    )

    print(
        f"Val F1        : {val_f1:.4f}"
    )

    print("-" * 60)

    if val_f1 > best_f1:

        best_f1 = val_f1
        best_epoch = epoch + 1

        model_path = os.path.join(
            MODEL_DIR,
            "ast_best.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            "Best AST model saved:"
        )

        print(model_path)


# =========================
# TEST
# =========================

print("\nLoading best AST model...")

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "ast_best.pth"
        ),
        map_location=DEVICE,
        weights_only=True
    )
)

print("\nEvaluating on test set...")

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1
) = evaluate(
    model,
    test_loader
)


# =========================
# FINAL RESULTS
# =========================

print("\n" + "=" * 60)
print("FINAL AST TEST RESULTS")
print("=" * 60)

print(
    f"Best Epoch : {best_epoch}"
)

print(
    f"Test Loss  : {test_loss:.4f}"
)

print(
    f"Accuracy   : {test_accuracy:.4f}"
)

print(
    f"Precision  : {test_precision:.4f}"
)

print(
    f"Recall     : {test_recall:.4f}"
)

print(
    f"F1 Score   : {test_f1:.4f}"
)

print("=" * 60)


# =========================
# SAVE RESULTS
# =========================

results = {
    "model": "AST",
    "dataset_size": {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset)
    },
    "best_epoch": best_epoch,
    "test_loss": test_loss,
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1_score": test_f1
}

results_path = os.path.join(
    RESULT_DIR,
    "ast_metrics.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nResults saved:")
print(results_path)

print("\nAST training complete.")