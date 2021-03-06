import time
import torch
import torch.nn as nn
from torchtext import data
from torch.autograd import Variable
from utils import subsequent_mask

class Batch():
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, src, trg = None, pad = 0):
        self.src      = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            self.trg      = trg[:, :-1]
            self.trg_y    = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens  = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    """
    Standard Training and Logging Function.
    """
    start        = time.time()
    total_tokens = 0
    total_loss   = 0
    tokens       = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src,
            batch.trg,
            batch.src_mask,
            batch.trg_mask
        )
        loss          = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss   += loss
        total_tokens += batch.ntokens
        tokens       += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print(f'Epoch Step: {i} '
                  f'Loss: {loss / batch.ntokens:.2f} '
                  f'Tokens per Sec: {tokens / elapsed:.2f}')
            start  = time.time()
            tokens = 0

    return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements     = count * max_src_in_batch
    tgt_elements     = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)

class SimpleLossCompute():
    """
    A simple loss compute and train function.
    """
    def __init__(self, generator, criterion, opt = None):
        self.generator = generator
        self.criterion = criterion
        self.opt       = opt

    def __call__(self, x, y, norm):
        x    = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(
                -1,
                x.size(-1)
            ),
            y.contiguous().view(-1)
        ) / norm
        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data[0] * norm

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(
                        d,
                        self.batch_size * 100
                ):
                    p_batch = data.batch(
                        sorted(
                            p,
                            key = self.sort_key
                        ),
                        self.batch_size,
                        self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(
                self.data(),
                self.random_shuffler
            )
        else:
            self.batches = []
            for b in data.batch(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn
            ):
                self.batches.append(
                    sorted(
                        b,
                        key = self.sort_key
                    )
                )

def rebatch(pad_idx, batch):
    """
    Fix order in torchtext to match ours.
    """
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)

    return Batch(src, trg, pad_idx)

class MultiGPULossCompute():
    """
    A multi-gpu loss compute and train function.
    """
    def __init__(self, generator, criterion, devices, opt = None, chunk_size = 5):
        # Send out to different gpus.
        self.generator  = generator
        self.criterion  = nn.parallel.replicate(
            criterion,
            devices = devices
        )
        self.opt        = opt
        self.devices    = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(
            self.generator,
            devices = self.devices
        )
        out_scatter = nn.parallel.scatter(
            out,
            target_gpus = self.devices
        )
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(
            targets,
            target_gpus = self.devices
        )

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions.
            out_column = [
                [
                    Variable(
                        o[:, i : i + chunk_size].data,
                        requires_grad = self.opt is not None
                    )
                    for o in out_scatter
                ]
            ]
            gen = nn.parallel.parallel_apply(
                generator,
                out_column
            )

            # Compute loss.
            y = [
                (
                    g.contiguous().view(-1, g.size(-1)),
                    t[:, i : i + chunk_size].contiguous().view(-1)
                )
                for g, t in zip(gen, targets)
            ]
            loss = nn.parallel.parallel_apply(
                self.criterion,
                y
            )

            # Sum and normalize loss.
            l = nn.parallel.gather(
                loss,
                target_device = self.devices[0]
            )
            l      = l.sum()[0] / normalize
            total += l.data[0]

            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(
                        out_column[j][0].grad.data.clone()
                    )

        if self.opt is not None:
            out_grad = [
                Variable(
                    torch.cat(
                        og,
                        dim = 1
                    )
                )
                for og in out_grad
            ]
            o1 = out
            o2 = nn.parallel.gather(
                out_grad,
                target_device = self.devices[0]
            )
            o1.backward(gradient = o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return total * normalize
