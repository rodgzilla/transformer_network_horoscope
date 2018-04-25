import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import datasets
from torchtext import data
from model_utils import make_model
from model_utils import greedy_decode
from optimizer import LabelSmoothing
from optimizer import NoamOpt
from train import MyIterator
from train import MultiGPULossCompute
from train import batch_size_fn
from train import run_epoch
from train import rebatch
from utils import subsequent_mask

spacy_en = spacy.load('en')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_file(fn):
    df       = pd.read_csv(fn)
    all_text = ' '.join(df.TEXT)

    return all_text

def preprocess_data(text):
    tok_text = tokenize_en(text)
    vocab    = set(tok_text)
    itos     = dict(enumerate(vocab))
    stoi     = {word : i for i, word in itos.items()}
    text_idx = [stoi[token] for token in tok_text]

    return text_idx, itos, stoi

def make_batch(text_idx, window_size):
    src = np.array([
        text_idx[i: i + window_size]
        for i in range(len(text_idx) - window_size - 1)
    ])
    tgt = np.array([
        text_idx[i: i + window_size]
        for i in range(1, len(text_idx) - window_size)
    ])
    shuffle_idx = np.random.permutation(src.shape[0])
    src         = src[idx]
    tgt         = tgt[idx]

    return src, tgt

class GenerationBatch():
    def __init__(self, src, trg = None, pad = 0):
        self.src      = src
        self.src_mask = self.make_std_mask(self.src, pad)

        if trg is not None:
            self.trg      = trg[:, :-1]
            self.trg_y    = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens  = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(data, pad):
        """
        Create a mask to hide padding and future words.
        """
        data_mask = (data != pad).unsqueeze(-2)
        data_mask = data_mask & Variable(subsequent_mask(data.size(-1)).type_as(data_mask.data))

        return data_mask

if __name__ == '__main__':
    BOS_WORD   = '<s>'
    EOS_WORD   = '</s>'
    BLANK_WORD = '<blank>'

    SRC = data.Field(
        sequential = True,
        tokenize   = tokenize_en,
        lower      = True,
        pad_token  = BLANK_WORD
    )
    TGT = data.Field(
        sequential = True,
        tokenize   = tokenize_en,
        lower      = True,
        init_token = BOS_WORD,
        eos_token  = EOS_WORD,
        pad_token  = BLANK_WORD
    )

    MAX_LEN = 500
    train = data.TabularDataset.splits(
        path = '../data',
        train = 'horoscope_2011.csv',
        format = 'csv',
        skip_header = True,
        fields = [
            ('SIGN', None),
            ('DATE', None),
            ('src', SRC)
        ]
    )[0]
    MIN_FREQ = 2
    SRC.build_vocab(train)
    TGT.build_vocab(train_y)

    devices = [0]
    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    print(
        len(SRC.vocab),
        len(TGT.vocab),
    )
    model = make_model(
        len(SRC.vocab),
        len(TGT.vocab),
        N = 6
    )
    model.cuda()
    criterion = LabelSmoothing(
        size        = len(TGT.vocab),
        padding_idx = pad_idx,
        smoothing   = 0.1
    )
    criterion.cuda()
    BATCH_SIZE = 500
    train_iter = MyIterator(
        train,
        batch_size = BATCH_SIZE,
        device = 0,
        repeat = False,
        sort_key = lambda x: (x.TEXT),
        batch_size_fn = batch_size_fn,
        train = True
    )
