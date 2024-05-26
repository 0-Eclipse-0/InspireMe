import torch

from model.EmotionsClassifier import *
from model.EmotionsDataset import *
from os import path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor():
    def __init__(self):
        self.dataset = EmotionsDataset("data/emotions.json")
        self.net = EmotionClassifier(len(self.dataset.vocab)).to(DEVICE)
        self.net.load_state_dict(torch.load(get_path("trained/model.pt"),
                                            map_location=torch.device(DEVICE)))
        self.net.eval()
        self.key = [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise"
        ]

    # Run prediction on string
    def predict(self, text):
        embedded = torch.tensor(pipeline(text, True, self.dataset.vocab)).to(DEVICE)

        with torch.no_grad():
            pred = self.net(embedded,
                            torch.zeros(1, dtype=torch.int).to(DEVICE)).argmax(dim=1).item()

        print("Your current emotional state: {}\n".format(self.key[pred]))

