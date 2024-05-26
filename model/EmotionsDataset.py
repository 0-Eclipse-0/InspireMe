import torchtext; torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
import torch
import warnings

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from os import path

# Consts
with warnings.catch_warnings(action="ignore"):  # Ignore 'en' warning
    TOKENIZER = get_tokenizer("spacy")
SPLIT = [.8, .1, .1]  # Train (80%), Validation (10%), Test (10%)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get relative path from root directory of project
def get_path(filename):
    return "/".join([*path.dirname(__file__).split('/')[:-1], *filename.split('/')])


# Embedding conversion pipeline (Label = False, Text = True)
def pipeline(data, type, vocab):
    if vocab is None: raise RuntimeError("Dataset not loaded...")
    return vocab(TOKENIZER(data)) if type else data

# collate_fn for DataLoader
def batchify(batch, vocab):
    labels, texts, offsets = [], [], [0]

    # Build batches as lists
    for label, text in batch:
        labels.append(pipeline(label, False, vocab))
        texts.append(torch.tensor(pipeline(text, True, vocab), dtype=torch.int64))
        offsets.append(texts[-1].size(0))

    # Convert lists to tensors
    labels = torch.tensor(labels, dtype=torch.int64).to(DEVICE)
    text = torch.cat(texts).to(DEVICE)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(DEVICE)

    return labels, text, offsets

# Parse Emotions dataset and add relevant data to our dataset
def clean_emotions(path_name):
    emotions_data = pd.read_json(get_path(path_name), lines=True)
    emotions_data = list(zip(emotions_data["label"], emotions_data["text"]))

    return emotions_data


class EmotionsDataset():
    def __init__(self, data_path):
        if int(sum(SPLIT)) != 1: raise ValueError("Data splits must sum to 1")

        self.emotions_data = clean_emotions(data_path)
        self.dataset_size = len(self.emotions_data)

        # Prepare vocabulary
        self.vocab = build_vocab_from_iterator([TOKENIZER(i[1]) for i in self.get_train()],
                                               specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
    # Don't worry yourself with these, they work, trust me
    # Return Training Set
    def get_train(self):
        return self.emotions_data[0:int(self.dataset_size * SPLIT[0])]

    # Return Testing Set
    def get_test(self):
        return self.emotions_data[int(self.dataset_size * SPLIT[0]):
                                  int(self.dataset_size * SPLIT[0] + self.dataset_size * SPLIT[1])]

    # Return validation set
    def get_val(self):
        return self.emotions_data[int(self.dataset_size * SPLIT[0] +
                                      self.dataset_size * SPLIT[1]):]


# Testing purposes
if __name__ == '__main__':
    pass
