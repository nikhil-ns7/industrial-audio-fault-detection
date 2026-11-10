import os
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf

from audio_preprocessing import preprocess_waveform


class MIMIIDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self.label_map = {
            "normal": 0,
            "abnormal": 1
        }

        self.raw_dir = r"C:\Fault_Recognition\data\raw"

    def __len__(self):
        return len(self.df)

    def load_audio(self, row):

        source = row["source"]
        file_path = row["file_path"]

        # -----------------------------------------
        # Extracted pump dataset
        # -----------------------------------------
        if source == "0_dB_pump":

            audio_path = file_path

            audio, sample_rate = sf.read(audio_path)

            waveform = torch.from_numpy(audio).float()

            if waveform.ndim == 2:
                waveform = waveform.T

            waveform = preprocess_waveform(
                waveform,
                sample_rate
            )

            return waveform

        # -----------------------------------------
        # ZIP datasets
        # -----------------------------------------
        else:

            zip_path = os.path.join(
                self.raw_dir,
                source
            )

            with zipfile.ZipFile(zip_path, "r") as z:

                with z.open(file_path) as audio_file:

                    audio_bytes = audio_file.read()

            # Write temporary WAV to memory
            import io

            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32"
            )

            waveform = torch.from_numpy(audio).float()

            if waveform.ndim == 2:
                waveform = waveform.T

            waveform = preprocess_waveform(
                waveform,
                sample_rate
            )

            return waveform

    def __getitem__(self, index):

        row = self.df.iloc[index]

        waveform = self.load_audio(row)

        label = self.label_map[row["label"]]

        return waveform, torch.tensor(label, dtype=torch.long)