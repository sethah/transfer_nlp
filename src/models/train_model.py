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


class AttentionIterator(object):
    def __init__(self, iterator, pos_start_index):
        self.iterator = iterator
        self.pos_start_index = pos_start_index

    def __iter__(self):
        for batch in self.iterator:
            # masked = AttentionIterator._mask(batch, self.mask_value)
            batch.text = batch.text.transpose(0, 1)
            batch_size, seq_len = batch.text.shape
            position_indices = torch.arange(self.pos_start_index, self.pos_start_index + seq_len,
                                            device=batch.text.device,
                                            dtype=torch.long).repeat(batch_size, 1)
            batch.text = torch.stack((batch.text, position_indices), dim=2)
            yield batch

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
            return self.generator(out.sum(dim=1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--special-tokens', type=str, default=None)
    parser.add_argument('--encoder-path', type=str)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--fix-length', type=int, default=200)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    train_device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    data_path = Path(args.data_path)
    save_path = data_path / "interim"
    model_path = Path(args.model_path)

    if args.special_tokens is not None:
        special_tokens = ",".split(args.special_tokens)
    else:
        special_tokens = ["<eos>", "<pad>"]
    vocab = load_vocab(special_tokens, args.encoder_path)
    pos_start = len(vocab)

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
    iters = {phase: data.BucketIterator(datasets[phase], repeat=False, batch_size=args.batch_size,
                                        train=phase == 'train', sort=False)
             for phase in datasets}
    att_iters = {phase: AttentionIterator(iters[phase], pos_start) for phase in datasets}

    embeds, encoder = load_model(model_path, len(special_tokens))

    def one_iter(model, criterion, optimizer, train=True):
        def _step(inp, targ):
            if train:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()
            output = model.forward(inp, torch.ones(1, inp.shape[1], inp.shape[1]).to(inp.device))
            loss = criterion(output, targ.unsqueeze(1).float() - 1.)
            if train:
                loss.backward()
                optimizer.step()
            return loss, output
        return _step

    def train_epoch(iter_step, loader, device, log_interval=100):
        def _step(epoch):
            run_loss = 0.
            run_correct = 0
            run_pos = 0
            run_count = 0
            for batch_idx, batch in enumerate(loader):
                inp, targ = batch.text.to(device), batch.label.to(device)
                loss, output = iter_step(inp, targ)
                preds = torch.sigmoid(output)
                run_correct += torch.sum((preds.squeeze() > 0.5).long() == (targ - 1)).item()
                run_pos += (targ - 1).sum().item()
                run_loss += loss.item()
                run_count += inp.shape[0]
                if (batch_idx + 1) % log_interval == 0:
                    samples_processed = batch_idx * inp.shape[0]
                    total_samples = len(loader.iterator.dataset)
                    logging.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},'
                                  'Pos: {}, Acc: {}'.format(
                        epoch, samples_processed, total_samples,
                        100. * batch_idx / len(loader.iterator), run_loss / run_count,
                    run_pos / run_count, run_correct / run_count))
                    run_pos = 0
                    run_count = 0
                    run_loss = 0
                    run_correct = 0
        return _step

    def valid_epoch(iter_step, loader, device):
        def _step(epoch):
            run_loss = 0.
            run_count = 0
            for batch_idx, batch in enumerate(loader):
                inp, targ = batch.text.to(device), batch.label.to(device)
                loss, output = iter_step(inp, targ)
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
    logging.debug(f"Training on {len(trainable_params)} parameters.")

    criterion = nn.BCEWithLogitsLoss().to(train_device)
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    step_iter = one_iter(model, criterion, optimizer, train=True)
    step_epoch = train_epoch(step_iter, att_iters['train'], train_device,
                             log_interval=args.log_interval)
    valid_step = one_iter(model, criterion, optimizer, train=False)
    validate = valid_epoch(valid_step, att_iters['test'], train_device)
    for i in range(args.num_epochs):
        step_epoch(i)
        validate(i)
