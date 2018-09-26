import torch


class AttentionIterator(object):
    """
    An iterator that adds positional indices to a sequence iterator
    """

    def __init__(self, iterator, pos_start_index):
        self.iterator = iterator
        # TODO: this leaks abstraction from elsewhere
        self.pos_start_index = pos_start_index

    def __iter__(self):
        for batch in self.iterator:
            batch.text = batch.text.transpose(0, 1)
            batch_size, seq_len = batch.text.shape
            position_indices = torch.arange(self.pos_start_index, self.pos_start_index + seq_len,
                                            device=batch.text.device,
                                            dtype=torch.long).repeat(batch_size, 1)
            batch.text = torch.stack((batch.text, position_indices), dim=2)
            yield batch

    def __len__(self):
        return len(self.iterator)


class Batch(object):

    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt, pad=0):
        self.src = src
        self.src_y = tgt
        self.src_mask = self.make_std_mask(self.src, pad)
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
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask.unsqueeze(0) == 0
