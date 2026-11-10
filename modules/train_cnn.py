import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cnn_dataset import CNNMIMIIDataset
from cnn_model import CNNFaultDetector


# =========================
# PATHS
# =========================

ROOT = r"C:\Fault_Recognition"

TRAIN_CSV = os.path.join(ROOT, "data", "splits", "train.csv")
VAL_CSV = os.path.join(ROOT, "data", "splits", "validation.csv")
TEST_CSV = os.path.join(ROOT, "data", "splits", "test.csv")

MODEL_DIR = os.path.join(ROOT, "model")
RESULT_DIR = os.path.join(ROOT, "result")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# SETTINGS
# =========================

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CNN INDUSTRIAL FAULT DETECTION")
print("=" * 60)
print("Device:", DEVICE)
print("Train CSV:", TRAIN_CSV)
print("Validation CSV:", VAL_CSV)
print("Test CSV:", TEST_CSV)


# =========================
# DATASETS
# =========================

print("\nLoading datasets...")

train_dataset = CNNMIMIIDataset(TRAIN_CSV)
val_dataset = CNNMIMIIDataset(VAL_CSV)
test_dataset = CNNMIMIIDataset(TEST_CSV)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# MODEL
# =========================

model = CNNFaultDetector().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# VALIDATION FUNCTION
# =========================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0

    with torch.no_grad():

        for mel, labels in loader:

            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(mel)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

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

    return avg_loss, accuracy, precision, recall, f1


# =========================
# TRAINING
# =========================

best_f1 = 0.0
best_epoch = 0

history = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_index, (mel, labels) in enumerate(train_loader):

        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(mel)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = total_loss / len(train_loader)

    val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(
        model,
        val_loader
    )

    print("\n" + "-" * 60)

    print(f"Epoch {epoch + 1}/{EPOCHS}")

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Accuracy  : {val_accuracy:.4f}")
    print(f"Val Precision : {val_precision:.4f}")
    print(f"Val Recall    : {val_recall:.4f}")
    print(f"Val F1        : {val_f1:.4f}")

    print("-" * 60)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

    # Save best model
    if val_f1 > best_f1:

        best_f1 = val_f1
        best_epoch = epoch + 1

        model_path = os.path.join(
            MODEL_DIR,
            "cnn_best.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(f"Best model saved: {model_path}")


# =========================
# LOAD BEST MODEL
# =========================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "cnn_best.pth"),
        map_location=DEVICE
    )
)


# =========================
# FINAL TEST
# =========================

print("\nEvaluating on test set...")

test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(
    model,
    test_loader
)


print("\n" + "=" * 60)
print("FINAL CNN TEST RESULTS")
print("=" * 60)

print(f"Best Epoch : {best_epoch}")
print(f"Test Loss  : {test_loss:.4f}")
print(f"Accuracy   : {test_accuracy:.4f}")
print(f"Precision  : {test_precision:.4f}")
print(f"Recall     : {test_recall:.4f}")
print(f"F1 Score   : {test_f1:.4f}")

print("=" * 60)


# =========================
# SAVE RESULTS
# =========================

results = {
    "model": "CNN",
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
    "cnn_metrics.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


history_path = os.path.join(
    RESULT_DIR,
    "cnn_training_history.json"
)

with open(history_path, "w") as f:
    json.dump(history, f, indent=4)


print("\nResults saved:")
print(results_path)
print(history_path)

print("\nCNN training complete.")