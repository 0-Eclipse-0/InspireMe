# Text Classifier
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform
from model.EmotionsDataset import *
from os import path, mkdir

# Consts
EMBED_SIZE = 256
NUM_CLASSES = 6
HIDDEN_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing network, to be changed to RNN or Transformer
class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size,
                                         EMBED_SIZE,
                                         sparse=False)
        self.fc1 = nn.Linear(in_features=EMBED_SIZE, out_features=HIDDEN_SIZE)
        self.dropout = nn.Dropout(p=0.25)
        self.GRU = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True, num_layers=2)
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        self.init_weights()

    def init_weights(self):
        weights = [-0.5, 0.5]
        self.embedding.weight.data.uniform_(*weights)
        self.fc1.weight.data.uniform_(*weights)
        self.fc2.weight.data.uniform_(*weights)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        out = self.fc1(out)
        out = self.dropout(out)
        out, _ = self.GRU(out)
        out = self.fc2(out)
        return out