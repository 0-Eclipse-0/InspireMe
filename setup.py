# Setup for backend of the program
# Notifies the user of any required steps before using InspireMe

import model.Trainer
import os
import gzip
import requests

from sys import exit
from tqdm import tqdm
from model.Predictor import Predictor
from utils.Color import Color

# Handle file control for eotions dataset
def download_emotions_dataset(color):
    response = None
    out_file = "data/emotions.json"
    if not os.path.exists("data"): os.makedirs("data")  # Make data directory

    # Download file from website
    for _ in tqdm(range(5), leave=False):
        response = requests.get("https://huggingface.co/datasets/dair-ai/emotion/resolve/main/data/data.jsonl.gz")

        if response.status_code != 200:
            print(color.error_tag, "Error downloading emotions dataset, check connection")
            exit(-1)

    # Unzip file
    print(color.print_tag("Decompressing..."))
    for _ in tqdm(range(2), leave=False):
        with open(out_file + ".gzip", "wb") as f:
            f.write(response.content)

        with gzip.open(out_file + ".gzip", "rb") as i:
            with open(out_file, "wb") as o:
                o.write(i.read())

        os.remove(out_file + ".gzip")

    print(color.success_tag("Dataset download successfully, saved to data/emotions.json"))

# Handle file control for the goemotions dataset
def download_go_emotions_dataset(color):
    response = None
    out_file = "data/emotions.json"
    if not os.path.exists("data"): os.makedirs("data")  # Make data directory

    # Download file from website
    for _ in tqdm(range(5), leave=False):
        response = requests.get("https://huggingface.co/datasets/dair-ai/emotion/resolve/main/data/data.jsonl.gz")

        if response.status_code != 200:
            print(color.error_tag, "Error downloading emotions dataset, check connection")
            exit(-1)


# Run model checks
def model_check(color):
    print('—' * 10, color.white_bold("Model Checks"), '—' * 10)

    # Check if pretrained model is available
    if os.path.exists("trained/model.pt"):
        print(color.success_tag("Model found in trained/model.pt"))
    else:
        do_train = input(color.print_tag("Model not found in trained/model.pt\n"
                                     f"\t {color.white_bold('1.)')} Train a model\n"
                                     f"\t {color.white_bold('2.)')} Download a pretrained model\n"
                                     f"{color.grey_bold('>>> ')}"))

        if do_train == '1':
            # Check if dataset is available
            if not os.path.exists("data/emotions.json"):
                print(color.print_tag("Emotions dataset missing, attempting to download..."))
                download_emotions_dataset(color)
            else:
                print(color.success_tag("Found dataset in data/emotions.json"))

            print(color.print_tag("Training model..."))
            model.Trainer.training_loop(color)
        elif do_train == '2':
            pass # Download a pretrained model TODO
        else:
            print(color.error_tag("Invalid choice, exiting..."))
            exit(-1)

if __name__ == "__main__":
    color = Color()
    model_check(color)
    predictor = Predictor()

    while True:
        statement = input("Enter a statement with what you're thinking: ")
        predictor.predict(statement)
