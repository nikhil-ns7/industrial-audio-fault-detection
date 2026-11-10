import torch
from transformers import AutoFeatureExtractor

from ast_model import ASTFaultDetector


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

print("=" * 60)
print("AST MODEL TEST")
print("=" * 60)

print("\nLoading AST feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Feature extractor loaded successfully.")

print("\nLoading AST model...")

model = ASTFaultDetector()

print("AST model loaded successfully.")

print("\nTesting dummy audio...")

# 10 seconds of 16 kHz audio
waveform = torch.randn(1, 160000)

inputs = feature_extractor(
    waveform.squeeze(0).numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

print("Input shape:", inputs["input_values"].shape)

with torch.no_grad():

    output = model(
        inputs["input_values"]
    )

print("Output shape:", output.shape)
print("Output:", output)

print("\nAST TEST SUCCESSFUL!")