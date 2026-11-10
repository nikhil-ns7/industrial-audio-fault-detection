# ============================================================
# INDUSTRIAL ACOUSTIC FAULT DETECTION
# ============================================================
#
# PAGE 1 : OVERVIEW
# PAGE 2 : PREDICTION
#
# MODELS:
#   CNN
#   AST
#   BEATs
#
# ============================================================

import os
import sys
import tempfile
import traceback

import streamlit as st
import torch
import torch.nn as nn
import torchaudio



# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Fault_Recognition"

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "model"
)

MODULES_DIR = os.path.join(
    PROJECT_ROOT,
    "modules"
)

if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial Acoustic Fault Detection",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

MODEL_METRICS = {

    "CNN": {
        "accuracy": 86.60,
        "precision": 94.37,
        "recall": 77.85,
        "f1": 85.32,

        "description":
            "CNN classification using Mel-spectrogram features.",

        "confusion": [
            [452, 22],
            [105, 369]
        ]
    },

    "AST": {
        "accuracy": 77.32,
        "precision": 76.16,
        "recall": 79.54,
        "f1": 77.81,

        "description":
            "Audio Spectrogram Transformer with a pretrained backbone.",

        "confusion": [
            [356, 118],
            [97, 377]
        ]
    },

    "BEATs": {
        "accuracy": 93.14,
        "precision": 93.00,
        "recall": 93.00,
        "f1": 93.15,

        "description":
            "Pretrained BEATs representation with Logistic Regression.",

        "confusion": [
            [442, 32],
            [33, 441]
        ]
    }
}


# ============================================================
# BEATs IMPORT
# ============================================================

BEATS_AVAILABLE = False

try:

    from beats_inference import predict_beats

    BEATS_AVAILABLE = True

except Exception:

    BEATS_AVAILABLE = False


# ============================================================
# CNN MODEL
# ============================================================
#
# IMPORTANT:
#
# The saved cnn_best.pth checkpoint showed:
#
# classifier.1.weight = [256, 4096]
#
# Therefore:
#
# classifier.0 = Dropout
# classifier.1 = Linear(4096, 256)
# classifier.2 = ReLU
# classifier.3 = Dropout
# classifier.4 = Linear(256, 2)
#
# ============================================================

class CNNFaultDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Fixed feature size
            # ------------------------------------------------

            nn.AdaptiveAvgPool2d(
                (4, 4)
            )
        )

        # ====================================================
        # CORRECT CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                2
            )
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# AST COMPATIBILITY
# ============================================================
# The local Windows Application Control policy blocks some native
# scikit-learn DLLs. Transformers imports roc_curve indirectly while
# loading AST. The AST model itself does not use roc_curve for inference,
# so provide a tiny compatibility stub before importing Transformers.
# ============================================================

def _prepare_ast_transformers_import():

    import types

    try:
        import sklearn.metrics  # noqa: F401
        return
    except Exception:
        metrics_stub = types.ModuleType("sklearn.metrics")

        def roc_curve(*args, **kwargs):
            raise RuntimeError(
                "roc_curve is not required for AST inference in this application."
            )

        metrics_stub.roc_curve = roc_curve

        sklearn_stub = types.ModuleType("sklearn")
        sklearn_stub.metrics = metrics_stub

        sys.modules.setdefault("sklearn", sklearn_stub)
        sys.modules["sklearn.metrics"] = metrics_stub


# ============================================================
# AST MODEL
# ============================================================

class ASTFaultDetector(nn.Module):

    def __init__(self):

        super().__init__()

        _prepare_ast_transformers_import()

        from transformers import ASTModel

        self.ast = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        # Freeze pretrained AST backbone

        for parameter in self.ast.parameters():

            parameter.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Linear(
                self.ast.config.hidden_size,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                2
            )
        )


    def forward(
        self,
        input_values
    ):

        outputs = self.ast(
            input_values=input_values
        )

        pooled = (
            outputs
            .last_hidden_state
            .mean(
                dim=1
            )
        )

        return self.classifier(
            pooled
        )


# ============================================================
# CHECKPOINT HELPER
# ============================================================

def get_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if "state_dict" in checkpoint:

            return checkpoint[
                "state_dict"
            ]

        if "model" in checkpoint:

            return checkpoint[
                "model"
            ]

    return checkpoint


# ============================================================
# LOAD CNN
# ============================================================

@st.cache_resource
def load_cnn_model():

    model_path = os.path.join(
        MODEL_DIR,
        "cnn_best.pth"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            "CNN checkpoint not found:\n"
            + model_path
        )

    model = CNNFaultDetector()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    state_dict = get_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# LOAD AST
# ============================================================

@st.cache_resource
def load_ast_model():

    model_path = os.path.join(
        MODEL_DIR,
        "ast_best.pth"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            "AST checkpoint not found:\n"
            + model_path
        )

    model = ASTFaultDetector()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    state_dict = get_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# LOAD AST PROCESSOR
# ============================================================

@st.cache_resource
def load_ast_processor():

    _prepare_ast_transformers_import()

    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )


# ============================================================
# CNN PREPROCESSING
# ============================================================

def preprocess_for_cnn(
    audio_path
):

    waveform, sample_rate = (
        torchaudio.load(
            audio_path
        )
    )

    # --------------------------------------------------------
    # Convert to mono
    # --------------------------------------------------------

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # --------------------------------------------------------
    # Resample to 16 kHz
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
    # Normalize amplitude
    # --------------------------------------------------------

    max_value = waveform.abs().max()

    if max_value > 0:

        waveform = (
            waveform / max_value
        )

    # --------------------------------------------------------
    # Mel Spectrogram
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

    # --------------------------------------------------------
    # Convert to dB
    # --------------------------------------------------------

    mel = (
        torchaudio.transforms
        .AmplitudeToDB(
            stype="power"
        )(mel)
    )

    # Current:
    #
    # [1, 128, time]
    #
    # Add batch:
    #
    # [1, 1, 128, time]

    mel = mel.unsqueeze(
        0
    )

    return mel


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_cnn(
    audio_path
):

    model = load_cnn_model()

    mel = preprocess_for_cnn(
        audio_path
    )

    mel = mel.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            mel
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_class = (
        torch.argmax(
            probabilities
        ).item()
    )

    normal_probability = (
        probabilities[0].item()
        * 100
    )

    abnormal_probability = (
        probabilities[1].item()
        * 100
    )

    if predicted_class == 0:

        label = "Normal"

        confidence = (
            normal_probability
        )

    else:

        label = "Abnormal"

        confidence = (
            abnormal_probability
        )

    return {

        "label": label,

        "confidence": confidence,

        "probabilities": {

            "Normal":
                normal_probability,

            "Abnormal":
                abnormal_probability
        }
    }


# ============================================================
# AST PREDICTION
# ============================================================

def predict_ast(
    audio_path
):

    model = load_ast_model()

    processor = load_ast_processor()

    waveform, sample_rate = (
        torchaudio.load(
            audio_path
        )
    )

    # --------------------------------------------------------
    # Mono
    # --------------------------------------------------------

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0
        )

    # --------------------------------------------------------
    # Resample
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
    # Convert to numpy
    # --------------------------------------------------------

    audio = waveform.numpy()

    # --------------------------------------------------------
    # AST preprocessing
    # --------------------------------------------------------

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = (
        inputs["input_values"]
        .to(
            DEVICE
        )
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_values
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_class = (
        torch.argmax(
            probabilities
        ).item()
    )

    normal_probability = (
        probabilities[0].item()
        * 100
    )

    abnormal_probability = (
        probabilities[1].item()
        * 100
    )

    if predicted_class == 0:

        label = "Normal"

        confidence = (
            normal_probability
        )

    else:

        label = "Abnormal"

        confidence = (
            abnormal_probability
        )

    return {

        "label": label,

        "confidence": confidence,

        "probabilities": {

            "Normal":
                normal_probability,

            "Abnormal":
                abnormal_probability
        }
    }


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def create_mel_spectrogram(
    audio_path
):

    waveform, sample_rate = (
        torchaudio.load(
            audio_path
        )
    )

    # Mono

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # 16 kHz

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

    mel_transform = (
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
    )

    mel = mel_transform(
        waveform
    )

    mel_db = (
        torchaudio.transforms
        .AmplitudeToDB(
            stype="power"
        )(mel)
    )

    return mel_db.squeeze(
        0
    ).numpy()


def show_mel_spectrogram(
    audio_path
):

    try:

        mel = create_mel_spectrogram(
            audio_path
        )

        # Normalize the Mel-spectrogram to 0-255 so Streamlit
        # can display it directly without Matplotlib.
        mel_min = mel.min()
        mel_max = mel.max()

        if mel_max > mel_min:
            image = (
                (mel - mel_min)
                / (mel_max - mel_min)
                * 255
            ).astype("uint8")
        else:
            image = (
                (mel * 0)
            ).astype("uint8")

        st.image(
            image,
            caption="Mel-Spectrogram",
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Could not generate spectrogram: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🔊 Fault Detection"
    )

    st.caption(
        "Industrial Audio Intelligence"
    )

    st.divider()

    st.subheader(
        "Navigation"
    )

    page = st.radio(
        "Select page",
        [
            "📊 Overview",
            "🔮 Prediction"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    st.caption(
        "Models"
    )

    st.write(
        "CNN"
    )

    st.write(
        "AST"
    )

    st.write(
        "BEATs"
    )

    st.divider()

    st.caption(
        "Dataset"
    )

    st.write(
        "MIMII Dataset"
    )


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.title(
        "Industrial Acoustic Fault Detection"
    )

    st.caption(
        "Intelligent machine-condition monitoring using acoustic "
        "signals, deep learning and pretrained audio representations."
    )

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader(
        "Project Objective"
    )

    st.info(
        "Detect whether an industrial machine is operating under "
        "Normal or Abnormal conditions from its acoustic signal, "
        "and compare three machine-learning approaches: CNN, AST and BEATs."
    )

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Executive Summary"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Best Accuracy",
            "93.14%",
            "BEATs"
        )

    with c2:

        st.metric(
            "Best F1",
            "93.15%",
            "BEATs"
        )

    with c3:

        st.metric(
            "Test Samples",
            "948"
        )

    with c4:

        st.metric(
            "Models Evaluated",
            "3"
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    cnn_col, ast_col, beats_col = st.columns(3)

    # CNN

    with cnn_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "CNN"
            )

            st.caption(
                MODEL_METRICS["CNN"]["description"]
            )

            st.metric(
                "Accuracy",
                "86.60%"
            )

            st.write(
                "Precision: **94.37%**"
            )

            st.write(
                "Recall: **77.85%**"
            )

            st.write(
                "Abnormal F1: **85.32%**"
            )

    # AST

    with ast_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "AST"
            )

            st.caption(
                MODEL_METRICS["AST"]["description"]
            )

            st.metric(
                "Accuracy",
                "77.32%"
            )

            st.write(
                "Precision: **76.16%**"
            )

            st.write(
                "Recall: **79.54%**"
            )

            st.write(
                "Abnormal F1: **77.81%**"
            )

    # BEATs

    with beats_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "🏆 BEATs"
            )

            st.caption(
                MODEL_METRICS["BEATs"]["description"]
            )

            st.metric(
                "Accuracy",
                "93.14%"
            )

            st.write(
                "Precision: **93.00%**"
            )

            st.write(
                "Recall: **93.00%**"
            )

            st.write(
                "Abnormal F1: **93.15%**"
            )

            st.success(
                "Best performing model"
            )

    # --------------------------------------------------------
    # ACCURACY COMPARISON
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Accuracy Comparison"
    )

    chart_col, metric_col = st.columns(
        [1.5, 1]
    )

    with chart_col:

        st.caption(
            "Test Accuracy Comparison"
        )

        acc1, acc2, acc3 = st.columns(3)

        with acc1:
            st.metric("CNN", "86.60%")

        with acc2:
            st.metric("AST", "77.32%")

        with acc3:
            st.metric("BEATs", "93.14%")

    with metric_col:

        st.subheader(
            "Detailed Metrics"
        )

        overview_model = st.selectbox(
            "Model",
            [
                "BEATs",
                "CNN",
                "AST"
            ],
            key="overview_model"
        )

        metrics = MODEL_METRICS[
            overview_model
        ]

        m1, m2 = st.columns(2)

        with m1:

            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.2f}%"
            )

        with m2:

            st.metric(
                "Precision",
                f"{metrics['precision']:.2f}%"
            )

        m3, m4 = st.columns(2)

        with m3:

            st.metric(
                "Recall",
                f"{metrics['recall']:.2f}%"
            )

        with m4:

            st.metric(
                "F1",
                f"{metrics['f1']:.2f}%"
            )

    # --------------------------------------------------------
    # CONFUSION MATRIX + FINDINGS
    # --------------------------------------------------------

    st.divider()

    cm_col, finding_col = st.columns(
        [1, 1.25]
    )

    with cm_col:

        st.subheader(
            "Confusion Matrix"
        )

        cm_model = st.selectbox(
            "Model",
            [
                "BEATs",
                "CNN",
                "AST"
            ],
            key="confusion_model"
        )

        cm = MODEL_METRICS[
            cm_model
        ]["confusion"]

        confusion_table = {
            "Actual": [
                "Normal",
                "Abnormal"
            ],
            "Predicted Normal": [
                cm[0][0],
                cm[1][0]
            ],
            "Predicted Abnormal": [
                cm[0][1],
                cm[1][1]
            ]
        }

        st.markdown(
            f"""
| Actual | Predicted Normal | Predicted Abnormal |
|---|---:|---:|
| Normal | **{cm[0][0]}** | **{cm[0][1]}** |
| Abnormal | **{cm[1][0]}** | **{cm[1][1]}** |
"""
        )

        st.caption(
            "Rows = Actual class • Columns = Predicted class"
        )

    with finding_col:

        st.subheader(
            "Key Findings"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                "**01 — BEATs achieved the strongest performance**"
            )

            st.write(
                "BEATs achieved 93.14% test accuracy and "
                "93.15% abnormal-class F1."
            )

            st.markdown(
                "**02 — CNN provides a strong baseline**"
            )

            st.write(
                "CNN achieved 86.60% test accuracy using "
                "Mel-spectrogram features."
            )

            st.markdown(
                "**03 — AST performed lower in this implementation**"
            )

            st.write(
                "AST achieved 77.32% test accuracy with the "
                "implemented frozen-backbone approach."
            )

            st.markdown(
                "**04 — BEATs provides the strongest representation**"
            )

            st.write(
                "The pretrained BEATs representation produced "
                "the best overall classification performance."
            )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "System Workflow"
    )

    w1, w2, w3, w4 = st.columns(4)

    with w1:

        with st.container(
            border=True
        ):

            st.subheader(
                "01"
            )

            st.write(
                "**Audio Input**"
            )

            st.caption(
                "Industrial machine audio"
            )

    with w2:

        with st.container(
            border=True
        ):

            st.subheader(
                "02"
            )

            st.write(
                "**Preprocessing**"
            )

            st.caption(
                "Mono conversion and 16 kHz"
            )

    with w3:

        with st.container(
            border=True
        ):

            st.subheader(
                "03"
            )

            st.write(
                "**Model Analysis**"
            )

            st.caption(
                "CNN / AST / BEATs"
            )

    with w4:

        with st.container(
            border=True
        ):

            st.subheader(
                "04"
            )

            st.write(
                "**Diagnosis**"
            )

            st.caption(
                "Normal / Abnormal"
            )


# ============================================================
# PAGE 2 — PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title(
        "Audio Fault Prediction"
    )

    st.caption(
        "Upload an industrial machine recording and classify "
        "its operating condition using a trained model."
    )

    st.divider()

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.subheader(
        "1. Upload Audio"
    )

    uploaded_file = st.file_uploader(
        "Choose a WAV file",
        type=["wav"]
    )

    if uploaded_file is None:

        st.info(
            "Upload a WAV file to begin prediction."
        )

        st.stop()

    # --------------------------------------------------------
    # TEMPORARY FILE
    # --------------------------------------------------------

    temp_audio_path = os.path.join(
        tempfile.gettempdir(),
        "fault_detection_uploaded.wav"
    )

    with open(
        temp_audio_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # --------------------------------------------------------
    # AUDIO INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "2. Audio Information"
    )

    try:

        waveform, sample_rate = (
            torchaudio.load(
                temp_audio_path
            )
        )

        channels = waveform.shape[0]

        samples = waveform.shape[1]

        duration = (
            samples / sample_rate
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.metric(
                "Sample Rate",
                f"{sample_rate / 1000:.1f} kHz"
            )

        with a2:

            st.metric(
                "Channels",
                channels
            )

        with a3:

            st.metric(
                "Duration",
                f"{duration:.2f} sec"
            )

        with a4:

            st.metric(
                "Samples",
                f"{samples:,}"
            )

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

    except Exception as e:

        st.error(
            f"Unable to read audio file: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "3. Select Model"
    )

    selected_model = st.radio(
        "Model",
        [
            "CNN",
            "AST",
            "BEATs"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    model_info = MODEL_METRICS[
        selected_model
    ]

    with st.container(
        border=True
    ):

        st.subheader(
            selected_model
        )

        st.caption(
            model_info["description"]
        )

        st.write(
            f"Test Accuracy: "
            f"**{model_info['accuracy']:.2f}%**"
        )

    # --------------------------------------------------------
    # AUDIO REPRESENTATION
    # --------------------------------------------------------

    st.subheader(
        "4. Audio Representation"
    )

    show_mel_spectrogram(
        temp_audio_path
    )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "5. Diagnosis"
    )

    run_prediction = st.button(
        "🔍 Run Prediction",
        type="primary",
        use_container_width=True
    )

    if not run_prediction:

        st.info(
            "Click Run Prediction to analyze the uploaded audio."
        )

        st.stop()


    # ========================================================
    # CNN
    # ========================================================

    if selected_model == "CNN":

        try:

            with st.spinner(
                "Running CNN..."
            ):

                result = predict_cnn(
                    temp_audio_path
                )

            label = result[
                "label"
            ]

            confidence = result[
                "confidence"
            ]

            normal_probability = result[
                "probabilities"
            ]["Normal"]

            abnormal_probability = result[
                "probabilities"
            ]["Abnormal"]

            # ------------------------------------------------
            # DIAGNOSIS
            # ------------------------------------------------

            if label == "Normal":

                st.success(
                    f"✓ NORMAL — Confidence: "
                    f"{confidence:.2f}%"
                )

            else:

                st.error(
                    f"⚠ ABNORMAL — Confidence: "
                    f"{confidence:.2f}%"
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Prediction Probabilities"
            )

            p1, p2 = st.columns(2)

            with p1:

                st.metric(
                    "Normal",
                    f"{normal_probability:.2f}%"
                )

            with p2:

                st.metric(
                    "Abnormal",
                    f"{abnormal_probability:.2f}%"
                )

            # ------------------------------------------------
            # MODEL PERFORMANCE
            # ------------------------------------------------

            st.subheader(
                "CNN Performance"
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "Test Accuracy",
                    "86.60%"
                )

            with r2:

                st.metric(
                    "Precision",
                    "94.37%"
                )

            with r3:

                st.metric(
                    "Abnormal F1",
                    "85.32%"
                )

        except Exception:

            st.error(
                "CNN prediction failed."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    traceback.format_exc()
                )


    # ========================================================
    # AST
    # ========================================================

    elif selected_model == "AST":

        try:

            with st.spinner(
                "Running AST..."
            ):

                result = predict_ast(
                    temp_audio_path
                )

            label = result[
                "label"
            ]

            confidence = result[
                "confidence"
            ]

            normal_probability = result[
                "probabilities"
            ]["Normal"]

            abnormal_probability = result[
                "probabilities"
            ]["Abnormal"]

            # ------------------------------------------------
            # DIAGNOSIS
            # ------------------------------------------------

            if label == "Normal":

                st.success(
                    f"✓ NORMAL — Confidence: "
                    f"{confidence:.2f}%"
                )

            else:

                st.error(
                    f"⚠ ABNORMAL — Confidence: "
                    f"{confidence:.2f}%"
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Prediction Probabilities"
            )

            p1, p2 = st.columns(2)

            with p1:

                st.metric(
                    "Normal",
                    f"{normal_probability:.2f}%"
                )

            with p2:

                st.metric(
                    "Abnormal",
                    f"{abnormal_probability:.2f}%"
                )

            # ------------------------------------------------
            # MODEL PERFORMANCE
            # ------------------------------------------------

            st.subheader(
                "AST Performance"
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "Test Accuracy",
                    "77.32%"
                )

            with r2:

                st.metric(
                    "Precision",
                    "76.16%"
                )

            with r3:

                st.metric(
                    "Abnormal F1",
                    "77.81%"
                )

        except Exception:

            st.error(
                "AST prediction failed."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    traceback.format_exc()
                )


    # ========================================================
    # BEATs
    # ========================================================

    elif selected_model == "BEATs":

        if not BEATS_AVAILABLE:

            st.error(
                "BEATs inference module could not be loaded."
            )

            st.code(
                r"C:\Fault_Recognition\modules\beats_inference.py"
            )

            st.stop()

        try:

            with st.spinner(
                "Running BEATs..."
            ):

                beats_result = predict_beats(
                    temp_audio_path
                )

            prediction = beats_result.get(
                "prediction",
                "Unknown"
            )

            confidence = float(
                beats_result.get(
                    "confidence",
                    0
                )
            )

            normal_probability = float(
                beats_result.get(
                    "normal_probability",
                    0
                )
            )

            abnormal_probability = float(
                beats_result.get(
                    "abnormal_probability",
                    0
                )
            )

            # ------------------------------------------------
            # DIAGNOSIS
            # ------------------------------------------------

            if prediction.lower() == "normal":

                st.success(
                    f"✓ NORMAL — Confidence: "
                    f"{confidence:.2f}%"
                )

            else:

                st.error(
                    f"⚠ ABNORMAL — Confidence: "
                    f"{confidence:.2f}%"
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "Prediction Probabilities"
            )

            p1, p2 = st.columns(2)

            with p1:

                st.metric(
                    "Normal",
                    f"{normal_probability:.2f}%"
                )

            with p2:

                st.metric(
                    "Abnormal",
                    f"{abnormal_probability:.2f}%"
                )

            # ------------------------------------------------
            # PERFORMANCE
            # ------------------------------------------------

            st.subheader(
                "BEATs Performance"
            )

            b1, b2, b3 = st.columns(3)

            with b1:

                st.metric(
                    "Test Accuracy",
                    "93.14%"
                )

            with b2:

                st.metric(
                    "Precision",
                    "93.00%"
                )

            with b3:

                st.metric(
                    "Abnormal F1",
                    "93.15%"
                )

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            st.subheader(
                "BEATs Pipeline"
            )

            st.info(
                "Audio → 16 kHz Mono → BEATs Encoder → "
                "768-dimensional embedding → Logistic Regression "
                "→ Normal / Abnormal"
            )

        except Exception:

            st.error(
                "BEATs prediction failed."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    traceback.format_exc()
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Industrial Acoustic Fault Detection • "
    "MIMII • CNN • AST • BEATs"
)