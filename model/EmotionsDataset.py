import torchtext; torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from os import path

# Consts
TOKENIZER = get_tokenizer("basic_english")
SPLIT = [.8, .1, .1]      # Train (80%), Validation (10%), Test (10%)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals (forgive me father for I have sinned)
glob_vocab = None

# Get relative path from root directory of project
def get_path(filename):
    return "/".join([*path.dirname(__file__).split('/')[:-1], *filename.split('/')])

# Embedding conversion pipeline (Label = False, Text = True)
def pipeline(data, type):
    if glob_vocab is None: raise RuntimeError("Dataset not loaded...")
    return glob_vocab(TOKENIZER(data)) if type else data

def get_vocab_size():
    return len(glob_vocab)

# collate_fn for DataLoader
def batchify(batch):
    labels, texts, offsets = [], [], [0]

    for label, text in batch:
        labels.append(pipeline(label, False))
        texts.append(torch.tensor(pipeline(text, True), dtype=torch.int64))
        offsets.append(texts[-1].size(0))

    labels = torch.tensor(labels, dtype=torch.int64)
    text = torch.cat(texts)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    return labels.to(DEVICE), text.to(DEVICE), offsets.to(DEVICE)

class EmotionsDataset():
    def __init__(self, path_name):
        if int(sum(SPLIT)) != 1: raise ValueError("Data splits must sum to 1")

        self.emotions_data = pd.read_json(get_path(path_name), lines=True)
        self.emotions_data = list(zip(self.emotions_data["label"][:1000], self.emotions_data["text"][:1000]))
        self.dataset_size = len(self.emotions_data)

        # Prepare vocabulary
        global glob_vocab
        glob_vocab = build_vocab_from_iterator([TOKENIZER(i[1]) for i in self.get_train()],
                                               specials=["<unk>"])
        glob_vocab.set_default_index(glob_vocab["<unk>"])

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
    emotions_dataset = EmotionsDataset("data/emotions.json")

    # dataloader = DataLoader(emotions_dataset.get_train(), batch_size=8, shuffle=False, collate_fn=batchify)
    print(pipeline("i beleive that i am much more sensitive to other peoples feelings and tend to be more compassionate", True))
