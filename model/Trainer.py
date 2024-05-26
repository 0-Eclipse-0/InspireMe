# Text Classifier
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform
from model.EmotionsDataset import *
from model.EmotionsClassifier import *
from os import path, mkdir

# Consts
EPOCHS = 4
BATCH = 12
EMBED_SIZE = 256
NUM_CLASSES = 6
HIDDEN_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def out_stats(iter, size, time, accuracy, loss, color):
    print(
        f"\r   {color.grey_bold('Iter:')} {iter}/{size}, "
        f"{color.grey_bold('Elapsed time:')} "
        f"{int(time)}s, "
        f"{color.grey_bold('Accuracy:')} {accuracy * 100:2.2f}%, "
        f"{color.grey_bold('Loss:')} {loss * BATCH:.6f}", end='')
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
        correct += (pred.argmax(dim=1) == label).sum().item()
        count += BATCH
        loss_total += loss
        optimizer.step()

        # Update stats every interval intervals
        if (idx) % interval == 0 or idx == len(dataloader):
            out_stats(idx, len(dataloader), time.time() - start_time, correct / count, loss_total / count, color)

    # Print final stats to account for off-by-one
    out_stats(len(dataloader), len(dataloader), time.time() - start_time, correct / count, loss_total / count, color)
    print()

# Validate model
def val_net(dataloader, net, color):
    accuracy, count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            pred = net(text, offsets)
            accuracy += (pred.argmax(dim=1) == label).sum().item()
            count += BATCH

    print(f"{color.grey_bold('Validation Accuracy:')} {accuracy / count * 100:2.2f}%\n".rjust(60))

# Test model
def test_net(dataloader, net, color):
    accuracy, count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            pred = net(text, offsets)
            accuracy += (pred.argmax(dim=1) == label).sum().item()
            count += BATCH

    print(f"{color.green_bold('Test Accuracy:')} {accuracy / count * 100:2.2f}%\n".rjust(57))

# Training a new model
def training_loop(color):
    data = EmotionsDataset("data/emotions.json")
    net = EmotionClassifier(len(data.vocab)).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Load datasets
    train_dataloader = DataLoader(
        data.get_train(),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=lambda batch: batchify(batch, data.vocab)
    )

    valid_dataloader = DataLoader(
        data.get_val(),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=lambda batch: batchify(batch, data.vocab)
    )

    test_dataloader = DataLoader(
        data.get_test(),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=lambda batch: batchify(batch, data.vocab)
    )

    # Run training loop
    for epoch in range(EPOCHS):
        print(color.white_bold(f"Epoch #{epoch + 1}").rjust(50))
        train_net(train_dataloader, net, optimizer, loss_fn, color)
        val_net(valid_dataloader, net, color)

    # Test our model
    test_net(test_dataloader, net, color)

    if not path.exists("trained"): mkdir("trained")
    torch.save(net.state_dict(), "trained/model.pt")

    print(color.success_tag("Saved trained model to trained/model.pt"))
