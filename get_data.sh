#!/bin/sh

# Download emotions dataset to data folder
get_emotions() {
  wget https://huggingface.co/datasets/dair-ai/emotion/resolve/main/data/data.jsonl.gz
  gzip -d data.jsonl.gz
  mv data.jsonl data/emotions.json
}

rm -rf data/*
get_emotions