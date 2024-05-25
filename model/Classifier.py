# Text Classifier
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform
from model.EmotionsDataset import *

# Consts
EPOCHS = 2
BATCH = 12
EMBED_SIZE = 256
NUM_CLASSES = 6
HIDDEN_SIZE = 128

# Globals
glob_model = None

# Testing network, to be changed to RNN or Transformer
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(get_vocab_size(),
                                         EMBED_SIZE,
                                         sparse=False)
        self.fc1 = nn.Linear(in_features=EMBED_SIZE, out_features=HIDDEN_SIZE)
        self.dropout = nn.Dropout(p=0.5)
        self.LSTM = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True, num_layers=2)
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
        out, _ = self.LSTM(out)
        out = self.fc2(out)
        return out

# Train model
def train_net(dataloader, net, optimizer, loss_fn, color):
    net.train()
    correct, count, loss_total, interval = 0, 0, 0, 10
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = net(text, offsets)
        loss = loss_fn(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        correct += (pred.argmax(1) == label).sum().item()
        count += BATCH
        loss_total += loss
        optimizer.step()

        if (idx + 1) % interval == 0:   # Update stats every 100 intervals
            elapsed = time.time() - start_time
            print(f"\r\t{color.grey_bold('Elapsed time:')} {int(elapsed)}s, "
                  f"{color.grey_bold('Accuracy:')} {correct / count * 100:2.2f}%, "
                  f"{color.grey_bold('Loss:')} {loss_total / count * BATCH:.6f}", end='')

    print() # Newline

# Validate model
def val_net(dataloader, net, color):
    accuracy, count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            pred = net(text, offsets)
            accuracy += (pred.argmax(1) == label).sum().item()
            count += BATCH

    print(f"{color.grey_bold('Validation Accuracy:')} {accuracy / count * 100:2.2f}%\n".rjust(60))

# Test model
def test_net(dataloader, net, color):
    accuracy, count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            pred = net(text, offsets)
            accuracy += (pred.argmax(1) == label).sum().item()
            count += BATCH

    print(f"{color.green_bold('Final Test Accuracy:')} {accuracy / count * 100:2.2f}%\n".rjust(60))

# Training a new model
def training_loop(color):
    data = EmotionsDataset("data/emotions.json")
    net = EmotionClassifier()
    optimizer = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Load datasets
    train_dataloader = DataLoader(
        data.get_train(),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=batchify
    )

    valid_dataloader = DataLoader(
        data.get_val(),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=batchify
    )

    test_dataloader = DataLoader(
        data.get_test(),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=batchify
    )

    # Run training loop
    for epoch in range(EPOCHS):
        print(color.white_bold(f"Epoch #{epoch + 1}").rjust(50))
        train_net(train_dataloader, net, optimizer, loss_fn, color)
        val_net(valid_dataloader, net, color)

    # Test our model
    test_net(test_dataloader, net, color)