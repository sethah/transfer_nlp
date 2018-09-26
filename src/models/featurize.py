import numpy as np
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from logging.config import fileConfig


import torch
import torch.nn as nn
from torchtext import data

from transformer.load import load_vocab, load_model

from src.text_utils import AttentionIterator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--special-tokens', type=str, default=None)
    parser.add_argument('--encoder-path', type=str)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--fix-length', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    train_device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    data_path = Path(args.data_path)
    save_path = data_path / "interim"
    model_path = Path(args.model_path)
    phases = []
    for phase in ['train', 'test']:
        if (save_path / f"features_{phase}_{args.layer}.npy").exists():
            logging.debug(f"Skipping featurize for phase: {phase}")
        else:
            phases.append(phase)
    if len(phases) == 0:
        sys.exit(0)

    if args.special_tokens is not None:
        special_tokens = ",".split(args.special_tokens)
    else:
        special_tokens = ["<eos>", "<pad>"]
    vocab = load_vocab(args.encoder_path, special_tokens)
    pos_start = len(vocab)

    TEXT = data.Field(lower=True, fix_length=args.fix_length)
    LABEL = data.Field(sequential=False, unk_token=None)
    TEXT.vocab = vocab

    idx_tokens = {phase: np.load(data_path / "interim" / f"bpe_{phase}_idx.npy")
                  for phase in phases}
    fields = [('text', TEXT), ('label', LABEL)]
    datasets = {phase: data.Dataset([data.Example.fromlist([toks, label], fields)
                        for toks, label in idx_tokens[phase]], fields) for phase in idx_tokens}
    LABEL.build_vocab(datasets['train'])
    iters = {phase: data.BucketIterator(datasets[phase], repeat=False, train=False,
                                        batch_size=args.batch_size, sort=False)
             for phase in datasets}
    att_iters = {phase: AttentionIterator(iters[phase], pos_start) for phase in datasets}

    embeds, encoder = load_model(model_path, len(special_tokens))

    class TruncEncoder(nn.Module):
        def __init__(self, embed, layers):
            super(TruncEncoder, self).__init__()
            self.layers = layers
            self.embed = embed

        def forward(self, x, mask):
            x = self.embed(x).sum(dim=2)
            for layer in self.layers:
                x = layer(x, mask)
            return x

    model = TruncEncoder(embeds, encoder.layers[:args.layer])

    for p in model.parameters():
        p.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    for phase, it in att_iters.items():
        preds = []
        targs = []
        for batch_idx, batch in tqdm(enumerate(it)):
            inp, targ = batch.text.to(train_device), batch.label.to(train_device)
            output = model.forward(inp, torch.ones(1, inp.shape[1], inp.shape[1]).to(inp.device))
            preds.append(output.detach().cpu().numpy())
            targs.append(targ.detach().cpu().numpy())
        preds_np = np.concatenate(preds, axis=0)
        targs_np = np.concatenate(targs, axis=0)
        np.save(save_path / f"features_{phase}_{args.layer}.npy", preds_np)
        np.save(save_path / f"labels_{phase}.npy", targs_np)

