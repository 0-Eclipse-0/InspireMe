# Main functions for data preprocessing
import pandas as pd

from os.path import isfile

# Load emotions dataset from file
def load_emotions():
    if not isfile("../data/emotions.csv"): raise FileNotFoundError("Emotions dataset"
                                                                   "missing from data/")
    emotions = pd.read_json("../data/emotions.json")