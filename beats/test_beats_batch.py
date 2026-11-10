import sys
import torch
import soundfile as sf
import joblib


# ============================================================
# PATHS
# ============================================================

BEATS_SOURCE = r"C:\Fault_Recognition\beats\source\beats"
CHECKPOINT = r"C:\Fault_Recognition\beats\BEATs_iter3_plus_AS2M.pt"
CLASSIFIER = r"C:\Fault_Recognition\model\beats_classifier.pkl"

DATA_ROOT = r"C:\Fault_Recognition\data\raw"


# ============================================================
# LOAD BEATs
# ============================================================

sys.path.insert(0, BEATS_SOURCE)

from BEATs import BEATs, BEATsConfig


print("=" * 65)
print("LOADING BEATs")
print("=" * 65)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

cfg = BEATsConfig(checkpoint["cfg"])

beats = BEATs(cfg)

beats.load_state_dict(checkpoint["model"])

beats.eval()

print("BEATs loaded successfully!")
print("Hidden size:", cfg.encoder_embed_dim)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("\n" + "=" * 65)
print("LOADING CLASSIFIER")
print("=" * 65)

classifier = joblib.load(CLASSIFIER)

print("Classifier loaded!")
print("Classes:", classifier.classes_)


# ============================================================
# FILES TO TEST
# ============================================================

normal_files = [
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_00\normal\00000779.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_00\normal\00000780.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_00\normal\00000781.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_00\normal\00000782.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_00\normal\00000783.wav",
]

abnormal_files = [
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_02\abnormal\00000000.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_02\abnormal\00000001.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_02\abnormal\00000002.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_02\abnormal\00000003.wav",
    r"C:\Fault_Recognition\data\raw\0_dB_pump\pump\id_02\abnormal\00000004.wav",
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_file(wav_path):

    # --------------------------------------------------------
    # Load audio exactly with soundfile
    # --------------------------------------------------------

    audio, sample_rate = sf.read(
        wav_path,
        dtype="float32"
    )

    waveform = torch.from_numpy(audio).float()

    # --------------------------------------------------------
    # Convert channels to mono
    # --------------------------------------------------------

    if waveform.ndim == 2:

        waveform = waveform.mean(dim=1)

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    waveform = waveform.unsqueeze(0)

    # --------------------------------------------------------
    # Resample if required
    # --------------------------------------------------------

    if sample_rate != 16000:

        import torchaudio

        resampler = torchaudio.transforms.Resample(
            sample_rate,
            16000
        )

        waveform = resampler(waveform)

    # --------------------------------------------------------
    # BEATs expects [batch, samples]
    # --------------------------------------------------------

    audio_input = waveform

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    with torch.no_grad():

        features = beats.extract_features(
            audio_input,
            padding_mask=None
        )[0]

    # --------------------------------------------------------
    # Mean pooling
    # --------------------------------------------------------

    embedding = features.mean(dim=1)

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    probabilities = classifier.predict_proba(
        embedding.numpy()
    )[0]

    prediction = classifier.predict(
        embedding.numpy()
    )[0]

    confidence = probabilities[prediction] * 100

    # 0 = normal
    # 1 = abnormal

    if prediction == 0:
        predicted_label = "NORMAL"
    else:
        predicted_label = "ABNORMAL"

    return predicted_label, confidence


# ============================================================
# TEST NORMAL FILES
# ============================================================

print("\n" + "=" * 65)
print("TESTING KNOWN NORMAL FILES")
print("=" * 65)

normal_correct = 0

for path in normal_files:

    predicted, confidence = predict_file(path)

    filename = path.split("\\")[-1]

    correct = predicted == "NORMAL"

    if correct:
        normal_correct += 1

    status = "✓ CORRECT" if correct else "✗ WRONG"

    print(
        f"{filename} | "
        f"Predicted: {predicted:<8} | "
        f"Confidence: {confidence:6.2f}% | "
        f"{status}"
    )


# ============================================================
# TEST ABNORMAL FILES
# ============================================================

print("\n" + "=" * 65)
print("TESTING KNOWN ABNORMAL FILES")
print("=" * 65)

abnormal_correct = 0

for path in abnormal_files:

    predicted, confidence = predict_file(path)

    filename = path.split("\\")[-1]

    correct = predicted == "ABNORMAL"

    if correct:
        abnormal_correct += 1

    status = "✓ CORRECT" if correct else "✗ WRONG"

    print(
        f"{filename} | "
        f"Predicted: {predicted:<8} | "
        f"Confidence: {confidence:6.2f}% | "
        f"{status}"
    )


# ============================================================
# SUMMARY
# ============================================================

total_correct = normal_correct + abnormal_correct
total_files = len(normal_files) + len(abnormal_files)

accuracy = (total_correct / total_files) * 100


print("\n" + "=" * 65)
print("BEATs BATCH TEST SUMMARY")
print("=" * 65)

print(
    f"Normal files:   "
    f"{normal_correct}/{len(normal_files)} correct"
)

print(
    f"Abnormal files: "
    f"{abnormal_correct}/{len(abnormal_files)} correct"
)

print(
    f"Overall:        "
    f"{total_correct}/{total_files} correct"
)

print(f"Accuracy:       {accuracy:.2f}%")

print("=" * 65)