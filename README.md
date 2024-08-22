# PredEmotions
A neural network dedicated to sentiment classification based on the underlying emotions
of a sentence.

## Usage
1. Install the requirements using `pip install -r requirements.txt`
2. Getting a model
    - Using a pretrained model: Move the model to `trained/model.pt` or use the existing model in the repository
    - Training a model:
       1. The dataset should placed in `data`.
       2. Run `python3 setupy.py`, it will explain the rest:
3. Running the classifier: Running the command `python3 setup.py` and hitting enter will start the inference loop.

### Datasets Being Used
- [Text to Emotion Classification](https://huggingface.co/datasets/dair-ai/emotion)
