from unicodedata import bidirectional
import torch
import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, n_feats, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.cnn = nn.Sequential(
            nn.Conv1d(
                n_feats,
                n_feats,
                10,
                2,
                padding=10//2
            ),
        )
        self.dense = nn.Sequential(
            nn.Linear(
                n_feats, 128
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            128,
            hidden_size,
            batch_first=True,
            bidirectional=False
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(
            hidden_size, output_size
        )
    
    def forward(self, x, hidden):
        x = x.squeeze(1)
        x = self.cnn(x)
        