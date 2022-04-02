import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import config, os
from torch.nn import functional as F
from sklearn import model_selection
from dataset import SpeechDataset
from model import SpeechModel
from train import *

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

cnn = nn.Sequential(
    nn.Conv1d(81, 81, 10, 2, padding=10//2),
    ActDropNormCNN1D(81, 0.2)
)
print(cnn(d).shape)