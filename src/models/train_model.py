import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import ftfy
import logging
from logging.config import fileConfig

import spacy

import torch
import torch.nn as nn
from torchtext import data, datasets

from transformer.load import load_vocab, load_model

import re
import html
re1 = re.compile(r'  +')



class AttentionIterator(object):
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        for batch in self.iterator:
            # masked = AttentionIterator._mask(batch, self.mask_value)
            batch.text = batch.text.transpose(0, 1)
            batch_size, seq_len = batch.text.shape
            position_indices = torch.arange(seq_len,
                                            device=batch.text.device,
                                            dtype=torch.long).repeat(batch_size, 1)
            batch.text = torch.stack((batch.text, position_indices), dim=2)
            yield batch


class Batch(object):
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, tgt, pad=0):
        self.src = src
        self.src_y = tgt
        self.src_mask = \
            self.make_std_mask(self.src, pad)
        self.ntokens = (self.src_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Batch.subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (size, size)
        subsequent_mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.uint8).view(1,
                                                                                                  size,
                                                                                                  size)
        return subsequent_mask == 0


def tokenizer(text):
    text = text.replace("<unk>", "unk")
    tokens = []
    for tok in nlp(text_standardize(ftfy.fix_text(text))):
        tokens.extend(text_encoder.bpe(tok.text).split(' '))
    return tokens

class EncoderOnly(nn.Module):
        def __init__(self, encoder, embed, generator):
            super(EncoderOnly, self).__init__()
            self.encoder = encoder
            self.embed = embed
            self.generator = generator

        def forward(self, src, src_mask):
            "Take in and process masked src and target sequences."
            embeds = self.embed(src)
            out = self.encoder.forward(embeds.sum(dim=2), src_mask)
            return self.generator(out[:, -1, :])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--special-tokens', type=str, default=None)
    parser.add_argument('--encoder-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--fix-length', type=int, default=200)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    train_device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    data_path = Path(args.data_path)
    save_path = data_path / "interim"

    if args.special_tokens is not None:
        special_tokens = ",".split(args.special_tokens)
    else:
        special_tokens = ["<eos>", "<pad>"]
    vocab = load_vocab(special_tokens, args.encoder_path)

    TEXT = data.Field(lower=True, fix_length=args.fix_length)
    LABEL = data.Field(sequential=False)
    TEXT.vocab = vocab

    phases = ['train', 'test']
    idx_tokens = {phase: np.load(data_path / "interim" / f"bpe_{phase}_idx.npy")
                  for phase in phases}
    fields = [('text', TEXT), ('label', LABEL)]
    datasets = {phase: data.Dataset([data.Example.fromlist([toks, label], fields)
                        for toks, label in idx_tokens[phase]], fields) for phase in idx_tokens}
    LABEL.build_vocab(datasets['train'])
    iters = {phase: data.BucketIterator(datasets[phase], repeat=False, batch_size=args.batch_size)
             for phase in datasets}
    att_iters = {phase: AttentionIterator(iters[phase]) for phase in datasets}

    embeds, encoder = load_model("/tmp/finetune-transformer-lm/model/")

    def one_iter(model, criterion, optimizer, train=True):
        def _step(inp, targ):
            if train:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()
            output = model.forward(inp, torch.ones(1, inp.shape[1], inp.shape[1]))
            loss = criterion(output, targ.unsqueeze(1).float() - 1.)
            if train:
                loss.backward()
                optimizer.step()
            return loss
        return _step

    def train_epoch(iter_step, loader, device, log_interval=100):
        def _step(epoch):
            run_loss = 0.
            run_count = 0
            for batch_idx, batch in enumerate(loader):
                inp, targ = batch.text.to(device), batch.label.to(device)
                loss = iter_step(inp, targ)
                run_loss += loss.item()
                run_count += inp.shape[0]
                if (batch_idx + 1) % log_interval == 0:
                    logging.debug(f"[Epoch {epoch}] Train loss: {run_loss / run_count}")
                    run_count = 0
                    run_loss = 0
        return _step

    def valid_epoch(iter_step, loader, device):
        def _step(epoch):
            run_loss = 0.
            run_count = 0
            for batch_idx, batch in enumerate(loader):
                inp, targ = batch.text.to(device), batch.label.to(device)
                loss = iter_step(inp, targ)
                run_loss += loss.item()
                run_count += inp.shape[0]
            logging.debug(f"Valid loss {run_loss / run_count}")
        return _step

    generator = nn.Linear(embeds.lut.embedding_dim, 1)
    model = EncoderOnly(encoder, embeds, generator).to(train_device)
    for p in model.embed.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.BCEWithLogitsLoss().to(train_device)
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)
    step_iter = one_iter(model, criterion, optimizer, train=True)
    step_epoch = train_epoch(step_iter, att_iters['train'], train_device, log_interval=1)
    valid_step = one_iter(model, criterion, optimizer, train=False)
    validate = valid_epoch(valid_step, att_iters['test'], train_device)
    for i in range(args.num_epochs):
        step_epoch(i)
        validate(i)
