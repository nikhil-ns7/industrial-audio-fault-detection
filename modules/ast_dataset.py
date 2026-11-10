import sys
import os
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_dataset import MIMIIDataset


class ASTMIMIIDataset(Dataset):

    def __init__(self, csv_path):
        self.dataset = MIMIIDataset(csv_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        waveform, label = self.dataset[index]

        return waveform, label