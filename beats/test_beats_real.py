import sys
import torch
import torchaudio
import joblib


# ============================================================
# PATHS
# ============================================================

BEATS_SOURCE = r"C:\Fault_Recognition\beats\source\beats"

CHECKPOINT = r"C:\Fault_Recognition\beats\BEATs_iter3_plus_AS2M.pt"

CLASSIFIER = r"C:\Fault_Recognition\model\beats_classifier.pkl"

WAV_FILE = (
    r"C:\Fault_Recognition\data\raw"
    r"\0_dB_pump\pump\id_00\normal\00000779.wav"
)


# ============================================================
# LOAD BEATs SOURCE
# ============================================================

sys.path.insert(0, BEATS_SOURCE)

from BEATs import BEATs, BEATsConfig


# ============================================================
# LOAD BEATs MODEL
# ============================================================

print("=" * 55)
print("LOADING BEATs")
print("=" * 55)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded.")

cfg = BEATsConfig(checkpoint["cfg"])

beats = BEATs(cfg)

beats.load_state_dict(checkpoint["model"])

beats.eval()

print("BEATs model loaded successfully!")
print("Hidden size:", cfg.encoder_embed_dim)


# ============================================================
# LOAD AUDIO
# ============================================================

print("\n" + "=" * 55)
print("LOADING AUDIO")
print("=" * 55)

print("File:")
print(WAV_FILE)

audio, sample_rate = torchaudio.load(WAV_FILE)

print("Original shape:", audio.shape)
print("Original sample rate:", sample_rate)


# ============================================================
# CONVERT MULTI-CHANNEL AUDIO TO MONO
# ============================================================

if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)

print("After mono:", audio.shape)


# ============================================================
# RESAMPLE TO 16 kHz IF NECESSARY
# ============================================================

if sample_rate != 16000:

    print(f"Resampling {sample_rate} Hz -> 16000 Hz")

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=16000
    )

    audio = resampler(audio)

else:

    print("Audio is already 16 kHz.")


print("After resampling:", audio.shape)


# ============================================================
# PREPARE AUDIO FOR BEATs
# ============================================================

audio = audio.squeeze(0)

audio = audio.unsqueeze(0)

print("BEATs input shape:", audio.shape)


# ============================================================
# EXTRACT BEATs FEATURES
# ============================================================

print("\n" + "=" * 55)
print("EXTRACTING BEATs FEATURES")
print("=" * 55)

with torch.no_grad():

    features = beats.extract_features(
        audio,
        padding_mask=None
    )[0]


print("Feature shape:", features.shape)


# ============================================================
# MEAN POOLING
# ============================================================

embedding = features.mean(dim=1)

print("Embedding shape:", embedding.shape)


# ============================================================
# CHECK EMBEDDING SIZE
# ============================================================

if embedding.shape[1] != 768:

    raise RuntimeError(
        f"Unexpected embedding size: {embedding.shape}. "
        "Expected [1, 768]."
    )

print("768-dimensional embedding confirmed.")


# ============================================================
# LOAD TRAINED CLASSIFIER
# ============================================================

print("\n" + "=" * 55)
print("LOADING CLASSIFIER")
print("=" * 55)

classifier = joblib.load(CLASSIFIER)

print("Classifier loaded successfully!")

print("Classifier classes:", classifier.classes_)

print("Classifier coefficient shape:", classifier.coef_.shape)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 55)
print("MAKING PREDICTION")
print("=" * 55)

embedding_np = embedding.numpy()

prediction = classifier.predict(embedding_np)[0]

probabilities = classifier.predict_proba(embedding_np)[0]

confidence = probabilities[prediction] * 100


# ============================================================
# CLASS LABEL
# ============================================================

if prediction == 0:

    label = "NORMAL"

else:

    label = "ABNORMAL"


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 55)
print("                BEATs DIAGNOSIS")
print("=" * 55)

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}%")

print("\nProbabilities:")

for class_index, probability in enumerate(probabilities):

    if class_index == 0:

        class_name = "NORMAL"

    else:

        class_name = "ABNORMAL"

    print(
        f"{class_name:<10}: "
        f"{probability * 100:.2f}%"
    )

print("=" * 55)

print("\nBEATs real-audio test completed successfully!")