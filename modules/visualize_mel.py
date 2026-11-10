import os
import webbrowser
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
import plotly.graph_objects as go


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_FILE = (
    r"C:\Fault_Recognition\data\raw\0_dB_pump"
    r"\pump\id_00\normal\00000779.wav"
)

RESULT_DIR = r"C:\Fault_Recognition\result"

PNG_OUTPUT = os.path.join(
    RESULT_DIR,
    "mel_spectrogram_normal.png"
)

HTML_OUTPUT = os.path.join(
    RESULT_DIR,
    "mel_spectrogram_normal.html"
)

SR = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128


# ============================================================
# MEL SCALE FUNCTIONS
# ============================================================

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank(sr, n_fft, n_mels):

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(
        low_mel,
        high_mel,
        n_mels + 2
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (n_fft + 1) * hz_points / sr
    ).astype(int)

    filterbank = np.zeros(
        (n_mels, n_fft // 2 + 1)
    )

    for m in range(1, n_mels + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:
            for k in range(left, center):
                filterbank[m - 1, k] = (
                    (k - left) /
                    (center - left)
                )

        if right > center:
            for k in range(center, right):
                filterbank[m - 1, k] = (
                    (right - k) /
                    (right - center)
                )

    return filterbank


# ============================================================
# LOAD AUDIO
# ============================================================

print("Loading MIMII recording...")

audio, sample_rate = sf.read(
    SAMPLE_FILE,
    dtype="float32"
)

print("Original sample rate:", sample_rate)

# 8 channels → mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)

if sample_rate != SR:
    raise ValueError(
        f"Expected {SR} Hz, got {sample_rate} Hz"
    )

# Normalize
max_value = np.max(np.abs(audio))

if max_value > 0:
    audio = audio / max_value


# ============================================================
# STFT
# ============================================================

print("Computing STFT...")

pad = N_FFT // 2

audio_padded = np.pad(
    audio,
    (pad, pad),
    mode="reflect"
)

n_frames = (
    1 +
    (len(audio_padded) - N_FFT)
    // HOP_LENGTH
)

window = np.hanning(N_FFT)

spectrogram = np.zeros(
    (N_FFT // 2 + 1, n_frames)
)

for i in range(n_frames):

    start = i * HOP_LENGTH

    frame = audio_padded[
        start:start + N_FFT
    ]

    frame = frame * window

    fft_result = np.fft.rfft(frame)

    spectrogram[:, i] = (
        np.abs(fft_result) ** 2
    )


# ============================================================
# MEL SPECTROGRAM
# ============================================================

print("Computing Mel spectrogram...")

mel_filterbank = create_mel_filterbank(
    SR,
    N_FFT,
    N_MELS
)

mel_spectrogram = (
    mel_filterbank @ spectrogram
)

mel_spectrogram = np.maximum(
    mel_spectrogram,
    1e-10
)

mel_db = 10 * np.log10(
    mel_spectrogram
)

# Relative dB scale
mel_db = mel_db - mel_db.max()

# Dynamic range
mel_db = np.clip(
    mel_db,
    -80,
    0
)


# ============================================================
# AXES
# ============================================================

time = (
    np.arange(n_frames)
    * HOP_LENGTH
    / SR
)

mel_axis = np.linspace(
    hz_to_mel(0),
    hz_to_mel(SR / 2),
    N_MELS
)


# ============================================================
# CREATE RESULT DIRECTORY
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 1. CREATE PNG
# ============================================================

print("Creating PNG...")

image_array = np.flipud(
    mel_db
)

# Convert -80...0 dB → 0...255
image_array = (
    (image_array + 80)
    / 80
    * 255
).astype(np.uint8)

image = Image.fromarray(
    image_array,
    mode="L"
).convert("RGB")

# Make it large
image = image.resize(
    (1400, 700)
)

draw = ImageDraw.Draw(image)

draw.text(
    (20, 15),
    "MIMII Pump - Log-Mel Spectrogram",
    fill="white"
)

draw.text(
    (20, 40),
    "Normal | id_00 | 00000779.wav",
    fill="white"
)

draw.text(
    (20, 660),
    "0 seconds",
    fill="white"
)

draw.text(
    (1320, 660),
    "10 seconds",
    fill="white"
)

image.save(
    PNG_OUTPUT
)

print("PNG saved.")


# ============================================================
# 2. CREATE INTERACTIVE GRAPH
# ============================================================

print("Creating interactive graph...")

fig = go.Figure(
    data=go.Heatmap(
        z=mel_db,
        x=time,
        y=mel_axis,
        colorscale="Viridis",
        zmin=-80,
        zmax=0,
        colorbar=dict(
            title="Amplitude (dB)"
        )
    )
)

fig.update_layout(

    title={
        "text":
        "MIMII Pump — Log-Mel Spectrogram"
        "<br><sup>Normal | id_00 | 00000779.wav</sup>"
    },

    xaxis=dict(
        title="Time (seconds)"
    ),

    yaxis=dict(
        title="Mel Frequency"
    ),

    width=1200,
    height=700,

    template="plotly_white"
)

fig.write_html(
    HTML_OUTPUT,
    include_plotlyjs=True
)

print("Interactive graph saved.")


# ============================================================
# OPEN INTERACTIVE GRAPH
# ============================================================

webbrowser.open(
    "file:///"
    + HTML_OUTPUT.replace("\\", "/")
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("MEL SPECTROGRAM VISUALIZATION COMPLETE")
print("=" * 60)

print("Mel shape:", mel_db.shape)

print()
print("PNG:")
print(PNG_OUTPUT)

print()
print("Interactive graph:")
print(HTML_OUTPUT)

print("=" * 60)