import argparse
from pathlib import Path

import spacy

import shutil

import torch
from torchtext import data, datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    sample_path = data_path / "sample"
    if sample_path.exists():
        shutil.rmtree(sample_path, ignore_errors=True)
    sub_dirs = list(data_path.iterdir())
    for sub_dir in sub_dirs:
        print(sample_path / sub_dir.stem)
        (sample_path / sub_dir.stem).mkdir(parents=True)
    for raw_file in (data_path / "raw").iterdir():
        shutil.copy(str(raw_file), str(sample_path / "raw"))
