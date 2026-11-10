import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go

from cnn_dataset import CNNMIMIIDataset
from cnn_model import CNNFaultDetector


# =========================
# PATHS
# =========================

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

RESULT_DIR = os.path.join(
    ROOT,
    "result"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# DEVICE
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CNN MODEL EVALUATION")
print("=" * 60)

print("Device:", DEVICE)
print("Test CSV:", TEST_CSV)
print("Model:", MODEL_PATH)


# =========================
# LOAD TEST DATASET
# =========================

print("\nLoading test dataset...")

test_dataset = CNNMIMIIDataset(TEST_CSV)

print("Test dataset size:", len(test_dataset))


test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# =========================
# LOAD MODEL
# =========================

print("\nLoading trained CNN...")

model = CNNFaultDetector().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("CNN model loaded successfully.")


# =========================
# PREDICTIONS
# =========================

print("\nGenerating predictions...")

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch_number, (mel, labels) in enumerate(test_loader):

        mel = mel.to(DEVICE)

        outputs = model(mel)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        if (batch_number + 1) % 20 == 0:
            print(
                f"Processed batch "
                f"{batch_number + 1}/{len(test_loader)}"
            )


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# =========================
# CLASSIFICATION REPORT
# =========================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=[
        "Normal",
        "Abnormal"
    ],
    digits=4
)

print(report)


# =========================
# SAVE CONFUSION MATRIX AS HTML
# =========================

print("\nCreating confusion matrix visualization...")

fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=[
            "Normal",
            "Abnormal"
        ],
        y=[
            "Normal",
            "Abnormal"
        ],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="CNN Test Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label"
)


# =========================
# SAVE FILE
# =========================

html_path = os.path.join(
    RESULT_DIR,
    "cnn_confusion_matrix.html"
)

fig.write_html(html_path)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("Confusion matrix saved to:")
print(html_path)