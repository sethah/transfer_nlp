import argparse
from pathlib import Path

import spacy

import torch
from torchtext import data, datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    args = parser.parse_args()

    spacy_en = spacy.load('en')
    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize=tokenizer)
    LABEL = data.Field(sequential=False)

    data_path = Path(args.data_path) / "aclImdb"
    train_ds = datasets.IMDB(str(data_path / "train"), TEXT, LABEL)
    test_ds = datasets.IMDB(str(data_path / "test"), TEXT, LABEL)

    # make splits for data
    # train, test = ds.split(split_ratio=0.7)