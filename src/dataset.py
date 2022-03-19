from codecs import ignore_errors
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import config

class MelSpectogram_log(nn.Module):
    def __init__(self, sample_rate, n_mels, win_length, hop_length):
        super(MelSpectogram_log, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels = self.n_mels,
            win_length = self.win_length,
            hop_length = self.hop_length
        )

    def forward(self, x):
        x = self.transform(x) # (channel, n_mels, time)
        x = np.log(x + 1e-14)
        return x        


class MFCCSPec(nn.Module):
    def __init__(self, sample_rate, n_mfcc):
        super(MFCCSPec, self).__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        self.transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc
        )

    def forward(self, x):
        x = self.transform(x) #(channel, n_mfcc, time)
        return x

class SpecAugment(nn.Module):
    def __init__(self, sample_rate, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        self.sample_rate = sample_rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
    
    def forward(self, x):
        return self.specaug(x)

# waverform, sr = torchaudio.load("/home/aditta/Desktop/ProjectT/input/cv-other-dev/sample-000000.mp3")
# ml = MFCCSPec(sample_rate=8000, n_mfcc=80)
# print(ml.forward(waverform).shape)

class AudioLabelTransform:
    def __init__(self):
        char_map = {
            "'": 0,
            '<SPACE>': 1,
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
            'e': 6,
            'f': 7,
            'g': 8,
            'h': 9,
            'i': 10,
            'j': 11,
            'k': 12,
            'l': 13,
            'm': 14,
            'n': 15,
            'o': 16,
            'p': 17,
            'q': 18,
            'r': 19,
            's': 20,
            't': 21,
            'u': 22,
            'v': 23,
            'w': 24,
            'x': 25,
            'y': 26,
            'z': 27
        }
        self.char_mapping = {}
        self.int_mapping = {}

        for data, lbl in char_map.items():
            self.char_mapping[data] = int(lbl)
            self.int_mapping[int(lbl)] = data
    
    def text_to_int(self, text):
        int_sequence = []
        for char in text.lower():
            if char == ' ':
                ch = self.char_mapping['<SPACE>']
            else:
                ch = self.char_mapping[char]
            int_sequence.append(ch)
        return int_sequence
    
    def int_to_text(self, lbl):
        sent = []
        for i in lbl:
            sent.append(self.int_mapping[i])
        return ''.join(sent).replace('<SPACE>', ' ')


# a = AudioLabelTransform().text_to_int("HELLO I am aditta das nishad")
# b = AudioLabelTransform().int_to_text(a)
# print(a)
# print(b)

class SpeechDataset(Dataset):
    def __init__(self, file, valid=False):
        super(SpeechDataset, self).__init__()
        self.file = file
        self.label = AudioLabelTransform()
        if valid:
            self.audio_transform = nn.Sequential(
                MelSpectogram_log(
                    sample_rate=8000,
                    n_mels=128,
                    win_length=160,
                    hop_length=80
                )
            )
        else:
            self.audio_transform = nn.Sequential(
                MelSpectogram_log(
                    sample_rate=8000,
                    n_mels=128,
                    win_length=160,
                    hop_length=80
                ),
                SpecAugment(
                    sample_rate=8000,
                    freq_mask=15,
                    time_mask=35
                )
            )

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        try:
            data = pd.read_csv(self.file)
            path = os.path.join(
                '/home/aditta/Desktop/ProjectT/input/',
                data["filename"].iloc[idx]
            )
            lbl_data = data["text"].iloc[idx]
            waveform, sr = torchaudio.load(path)
            aug = self.audio_transform(waveform)
            aug = aug.clone().detach()
            audiolbl = self.label.text_to_int(lbl_data)
            audiolbl = torch.tensor(audiolbl, dtype=torch.long)
            lbl_len = len(audiolbl)
            spec_len = aug.shape[-1] // 2
            
            if aug.shape[0] > 1:
                raise Exception(f'Dual Channel, skip this file: {path}')
            if aug.shape[2] > 1650:
                raise Exception(f'Spectogram is large: {aug.shape[2]}, {path}')
        except Exception as e:
            print(e)

        return {
            'spectogram': aug,
            'label': audiolbl,
            'lbl_length': lbl_len,
            'spec_len': spec_len
        }

def collate_fn(data):
    specs = []
    labels = []
    input_length, target_length = [], []
    for d in data:
        spec = d["spectogram"].to(config.device)
        label = d["label"].to(config.device)
        spec_len = d["spec_len"]
        lbl_length = d["lbl_length"]

        specs.append(spec.squeeze(0).tratranspose(0, 1))
        labels.append(label)
        input_length.append(spec_len)
        target_length.append(lbl_length)
    spec = nn.utils.rnn.pad_sequence(
        specs, batch_first=True
    ).unsqueeze(1).transpose(2, 3)
    label = nn.utils.rnn.pad_sequence(
        labels, batch_first=True
    )
    return specs, labels, torch.tensor(input_length, dtype=torch.long), torch.tensor(target_length, dtype=torch.long)

# from glob import glob
# from tqdm import tqdm
# file_path = glob("/home/aditta/Desktop/ProjectT/input/cv-other-dev/*")
# for idx, f in tqdm(enumerate(file_path), total=len(file_path)):
# a = SpeechDataset("/home/aditta/Desktop/ProjectT/input/cv-other-dev.csv").__getitem__(3)
# print(a)
# print(collate_fn(a))
