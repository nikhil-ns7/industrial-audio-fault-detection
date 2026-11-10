import sys
import os
import torch

# BEATs source location
BEATS_SOURCE = r"C:\Fault_Recognition\beats\source\beats"
sys.path.insert(0, BEATS_SOURCE)

from BEATs import BEATs, BEATsConfig


# --------------------------------------------------
# Paths
# --------------------------------------------------

CHECKPOINT = r"C:\Fault_Recognition\beats\BEATs_iter3_plus_AS2M.pt"


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

print("Loading BEATs checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded.")

cfg = BEATsConfig(checkpoint["cfg"])

model = BEATs(cfg)
model.load_state_dict(checkpoint["model"])
model.eval()

print("BEATs model loaded successfully!")
print("Hidden size:", cfg.encoder_embed_dim)


# --------------------------------------------------
# Test with dummy 16 kHz audio
# --------------------------------------------------

print("\nTesting feature extraction...")

# 10 seconds of silent 16 kHz mono audio
audio = torch.zeros(1, 16000 * 10)

padding_mask = torch.zeros(
    audio.shape,
    dtype=torch.bool
)

with torch.no_grad():

    features = model.extract_features(
        audio,
        padding_mask=padding_mask
    )[0]

print("Feature shape:", features.shape)

# Mean pool across time
embedding = features.mean(dim=1)

print("Embedding shape:", embedding.shape)

print("\nBEATs test completed successfully!")