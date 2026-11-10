import os
import sys

import torch
import torch.nn.functional as F
import torchaudio


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODULES_DIR = os.path.join(
    BASE_DIR,
    "modules"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)


# ============================================================
# IMPORT YOUR EXISTING MODELS
# ============================================================

from cnn_model import CNNFaultDetector
from ast_model import ASTFaultDetector

from transformers import AutoFeatureExtractor


# ============================================================
# MODEL PATHS
# ============================================================

CNN_PATH = os.path.join(
    MODEL_DIR,
    "cnn_best.pth"
)

AST_PATH = os.path.join(
    MODEL_DIR,
    "ast_best.pth"
)

AST_MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)


# ============================================================
# LOAD CNN
# ============================================================

def load_cnn():

    # IMPORTANT:
    # Your CNNFaultDetector does NOT accept num_classes.

    model = CNNFaultDetector()

    checkpoint = torch.load(
        CNN_PATH,
        map_location="cpu"
    )

    # Handle different checkpoint formats

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# LOAD AST
# ============================================================

def load_ast():

    # Your AST model already contains
    # the 2-class classifier.

    model = ASTFaultDetector()

    checkpoint = torch.load(
        AST_PATH,
        map_location="cpu"
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )

    model.eval()


    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            AST_MODEL_NAME
        )
    )

    return (
        model,
        feature_extractor
    )


# ============================================================
# CNN PREDICTION
# ============================================================

@torch.inference_mode()
def predict_cnn(
    audio,
    sample_rate,
    model
):

    # --------------------------------------------------------
    # MONO
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = audio.mean(
            axis=1
        )


    # --------------------------------------------------------
    # WAVEFORM
    # --------------------------------------------------------

    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)


    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    if sample_rate != 16000:

        resampler = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
        )

        waveform = resampler(
            waveform
        )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_value = waveform.abs().max()

    if max_value > 0:

        waveform = (
            waveform / max_value
        )


    # --------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------

    mel_transform = (
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
    )

    mel = mel_transform(
        waveform
    )


    mel = (
        torchaudio.transforms.AmplitudeToDB(
            stype="power"
        )(mel)
    )


    # --------------------------------------------------------
    # CNN INPUT
    # --------------------------------------------------------

    # [1, 128, time]
    # →
    # [1, 1, 128, time]

    if mel.dim() == 3:

        mel = mel.unsqueeze(1)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    logits = model(
        mel
    )


    probabilities = F.softmax(
        logits,
        dim=1
    )


    prediction = (
        probabilities
        .argmax(
            dim=1
        )
        .item()
    )


    confidence = (
        probabilities[
            0,
            prediction
        ].item()
        * 100
    )


    labels = [
        "Normal",
        "Abnormal"
    ]


    return {

        "label":
            labels[prediction],

        "confidence":
            confidence,

        "probabilities": {

            "Normal":
                probabilities[
                    0,
                    0
                ].item() * 100,

            "Abnormal":
                probabilities[
                    0,
                    1
                ].item() * 100
        }
    }


# ============================================================
# AST PREDICTION
# ============================================================

@torch.inference_mode()
def predict_ast(
    audio,
    sample_rate,
    model,
    feature_extractor
):

    # --------------------------------------------------------
    # MONO
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = audio.mean(
            axis=1
        )


    # --------------------------------------------------------
    # WAVEFORM
    # --------------------------------------------------------

    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    if sample_rate != 16000:

        resampler = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
        )

        waveform = resampler(
            waveform.unsqueeze(0)
        ).squeeze(0)


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_value = waveform.abs().max()

    if max_value > 0:

        waveform = (
            waveform / max_value
        )


    audio_array = (
        waveform
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # AST FEATURE EXTRACTION
    # --------------------------------------------------------

    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )


    input_values = (
        inputs["input_values"]
    )


    # --------------------------------------------------------
    # AST MODEL
    # --------------------------------------------------------

    logits = model(
        input_values
    )


    probabilities = F.softmax(
        logits,
        dim=1
    )


    prediction = (
        probabilities
        .argmax(
            dim=1
        )
        .item()
    )


    confidence = (
        probabilities[
            0,
            prediction
        ].item()
        * 100
    )


    labels = [
        "Normal",
        "Abnormal"
    ]


    return {

        "label":
            labels[prediction],

        "confidence":
            confidence,

        "probabilities": {

            "Normal":
                probabilities[
                    0,
                    0
                ].item() * 100,

            "Abnormal":
                probabilities[
                    0,
                    1
                ].item() * 100
        }
    }