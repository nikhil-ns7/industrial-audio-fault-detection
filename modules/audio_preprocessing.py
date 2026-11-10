import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def preprocess_waveform(waveform, sample_rate):
    """
    Common preprocessing for MIMII audio.

    Steps:
    1. Convert multi-channel audio to mono
    2. Resample to 16 kHz
    3. Convert to float32
    4. Normalize amplitude
    """

    # Convert multi-channel audio to mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    # Resample if required
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            TARGET_SAMPLE_RATE
        )

    # Float32
    waveform = waveform.float()

    # Amplitude normalization
    max_value = waveform.abs().max()

    if max_value > 0:
        waveform = waveform / max_value

    return waveform


def waveform_to_mel(
    waveform,
    sample_rate=TARGET_SAMPLE_RATE,
    n_mels=128,
    n_fft=1024,
    hop_length=512
):
    """
    Convert a waveform into a log-Mel spectrogram.
    """

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel = mel_transform(waveform)

    # Convert to decibel scale
    db_transform = torchaudio.transforms.AmplitudeToDB()

    mel_db = db_transform(mel)

    return mel_db