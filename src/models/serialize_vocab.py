import shutil
import numpy as np
import argparse
from pathlib import Path
import logging
import os
import sys
import subprocess
import functools

import spacy

from torchtext import data, datasets

from transformer.load import load_vocab

from src.text_utils import TextEncoder, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--special-tokens', type=str, default=None)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--temp-dir', type=str, default='/tmp/')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_path = data_path / "interim" / "vocab"
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    phases = [p for p in ['train', 'test'] if not (save_path / f"bpe_{p}_idx.npy").exists()]
    if len(phases) == 0:
        logging.debug(f"Skipping vocab serialization")
        sys.exit(0)

    encoder_path = model_path / "encoder_bpe_40000.json"
    bpe_path = model_path / "vocab_40000.bpe"
    if not model_path.exists():
        tmp_path = Path(args.temp_dir) / model_path.stem
        subprocess.call(["git", "clone", "https://github.com/openai/finetune-transformer-lm/",
                         str(tmp_path)])
        os.rename(str(tmp_path / "model"), model_path)
        shutil.rmtree(str(tmp_path))

    if args.special_tokens is not None:
        special_tokens = ",".split(args.special_tokens)
    else:
        special_tokens = special_tokens = ["_classify_", "<pad>"]
    vocab = load_vocab(str(encoder_path), special_tokens)

    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
    text_encoder = TextEncoder(str(encoder_path), str(bpe_path))

    tokenize = functools.partial(tokenizer, nlp=nlp, encoder=text_encoder)
    TEXT = data.Field(lower=True, tokenize=tokenize)
    LABEL = data.Field(sequential=False)
    TEXT.vocab = vocab

    datasets = {p: datasets.IMDB(str(data_path / "processed" / "aclImdb" / p), TEXT, LABEL)
                for p in phases}

    for phase, ds in datasets.items():
        phase_path = save_path / f"bpe_{phase}_idx.npy"
        if not phase_path.exists():
            docs = np.array([(ex.text, ex.label) for ex in ds.examples])
            np.save(phase_path, docs)

