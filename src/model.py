import torch
import torch.nn as nn

from torch.nn import functional as F

d = torch.randn(32, 81, 1594)
class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        # x = self.norm(self.dropout(F.gelu(x)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x

class SpeechModel(nn.Module):
    def __init__(self, hidden_size, num_layers, n_feats, dropout, num_classes):
        super(SpeechModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.n_feats = n_feats
        self.dropout = dropout
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_feats, self.n_feats, 10, padding=10//5),
            ActDropNormCNN1D(self.n_feats, dropout)
        )
        self.dense = nn.Sequential(
            nn.Linear(self.n_feats, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(self.hidden_size, self.num_classes)
    
    def _init_hidden(self, batch_size):
        n, hs = self.num_layer, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1) # bs, time, feature
        # print(x.shape)
        x = self.cnn(x) # bs, time, feature
        # print(x.shape)
        x = self.dense(x)  # bs, time, feature
        # print(x.shape)
        x = x.transpose(0, 1) # time, bs, feature
        # print(x.shape)
        out, (hn, cn) = self.lstm(x, hidden)
        # print(out.shape)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))
        x = self.final_fc(x)
        return x, (hn, cn)