import sys
import os
import torch
from torch.utils.data import Dataset

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from audio_dataset import MIMIIDataset
from audio_preprocessing import waveform_to_mel


class CNNMIMIIDataset(Dataset):

    def __init__(self, csv_path):
        self.dataset = MIMIIDataset(csv_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        waveform, label = self.dataset[index]

        # Convert waveform to log-Mel spectrogram
        mel = waveform_to_mel(
            waveform,
            sample_rate=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )

        # Add channel dimension for CNN
        # [128, 313] → [1, 128, 313]
        mel = mel.unsqueeze(0)

        return mel, label