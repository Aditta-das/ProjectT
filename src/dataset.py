from cProfile import label
from email.mime import audio
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from sklearn import model_selection
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


class SpeechDataset(Dataset):
    def __init__(self, json_path, valid=False):
        super(SpeechDataset, self).__init__()

        self.data = pd.read_json(json_path, lines=True)

        self.label = AudioLabelTransform()
        if valid:
            self.audio_transform = nn.Sequential(
                MelSpectogram_log(
                    sample_rate=8000,
                    n_mels=81,
                    win_length=160,
                    hop_length=80
                )
            )
        else:
            self.audio_transform = nn.Sequential(
                MelSpectogram_log(
                    sample_rate=8000,
                    n_mels=81,
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
        return len(self.data)

    def __getitem__(self, idx):
        try:
            filepath = self.data.path.iloc[idx]
            # print(path)
            lbl_data = self.data["text"].iloc[idx]
            waveform, sr = torchaudio.load(filepath)
            aug = self.audio_transform(waveform) # (channel, n_mels, time)
            aug = aug.clone().detach()
            audiolbl = self.label.text_to_int(lbl_data)
            audiolbl = torch.tensor(audiolbl, dtype=torch.long)
            # data["label"] = audiolbl
            lbl_len = len(audiolbl)
            spec_len = aug.shape[-1] // 2
            if spec_len < lbl_len:
                raise Exception('spectrogram len is bigger then label len')
            if aug.shape[0] > 1:
                raise Exception(f'Dual Channel, skip this file: {filepath}')
            if aug.shape[2] > 1650:
                raise Exception(f'Spectogram is large: {aug.shape[2]}, {filepath}')
        except Exception as e:
            print(e)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        return aug, audiolbl, spec_len, lbl_len




def collate_fn_padding(data):
    specs = []
    labels = []
    input_length = []
    label_length = []
    for (aug, audiolbl, spec_len, lbl_len) in data:
        if aug is None:
            continue
        # print(aug.squeeze(0).transpose(0, 1).shape)
        specs.append(aug.squeeze(0).transpose(0, 1)) # time, n_mels
        labels.append(audiolbl)
        input_length.append(spec_len)
        label_length.append(lbl_len)
    spectogram = nn.utils.rnn.pad_sequence(
        specs, batch_first=True
    ).unsqueeze(1).transpose(2, 3)
    # print(f"spec : {spectogram.shape}")
    labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True
    )
    input_length = input_length
    label_length = label_length
    return spectogram, labels, input_length, label_length


# a = SpeechDataset(os.path.join(config.json_data_path, "train.json")).__getitem__(1110)
# print(a[0].shape)
# print(a)