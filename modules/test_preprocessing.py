import os
import sys
import torch
import soundfile as sf

# Allow importing from modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_preprocessing import preprocess_waveform, waveform_to_mel


# Find one WAV from the extracted pump dataset
pump_root = r"C:\Fault_Recognition\data\raw\0_dB_pump\pump"

sample_file = None

for root, dirs, files in os.walk(pump_root):
    for file in files:
        if file.lower().endswith(".wav"):
            sample_file = os.path.join(root, file)
            break

    if sample_file:
        break


if sample_file is None:
    raise FileNotFoundError("No WAV file found.")


print("=" * 60)
print("PREPROCESSING TEST")
print("=" * 60)

print("\nFile:")
print(sample_file)


# Load audio
audio, sample_rate = sf.read(sample_file)

# soundfile returns:
# (samples, channels) for multi-channel audio
waveform = torch.from_numpy(audio).float()

# Convert channels × samples
if waveform.ndim == 2:
    waveform = waveform.T

print("\nOriginal audio:")
print("Shape:", waveform.shape)
print("Sample rate:", sample_rate)
print("Channels:", waveform.shape[0])


# Preprocess
processed = preprocess_waveform(
    waveform,
    sample_rate
)

print("\nAfter preprocessing:")
print("Shape:", processed.shape)
print("Sample rate:", 16000)
print("Duration:", processed.shape[-1] / 16000, "seconds")
print("Min:", processed.min().item())
print("Max:", processed.max().item())


# Mel spectrogram
mel = waveform_to_mel(processed)

print("\nMel spectrogram:")
print("Shape:", mel.shape)

print("\n" + "=" * 60)
print("PREPROCESSING TEST COMPLETE")
print("=" * 60)