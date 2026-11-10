import sys

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = r"C:\Fault_Recognition"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT BEATs INFERENCE
# ============================================================

from modules.beats_inference import predict_beats


# ============================================================
# TEST AUDIO
# ============================================================

WAV_FILE = (
    r"C:\Fault_Recognition\data\raw"
    r"\0_dB_pump\pump\id_02\abnormal\00000000.wav"
)


# ============================================================
# RUN PREDICTION
# ============================================================

print("=" * 55)
print("BEATs INFERENCE MODULE TEST")
print("=" * 55)

result = predict_beats(WAV_FILE)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 55)
print("RESULT")
print("=" * 55)

print("Prediction :", result["prediction"])
print(f"Confidence : {result['confidence']:.2f}%")

print(
    f"Normal     : "
    f"{result['normal_probability']:.2f}%"
)

print(
    f"Abnormal   : "
    f"{result['abnormal_probability']:.2f}%"
)

print(
    "Embedding  :",
    result["embedding_shape"]
)

print("=" * 55)
print("\nBEATs inference module test completed!")