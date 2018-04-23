import copy
from utils import clones
import torch.nn as nn
from layers import MultiHeadedAttention
from layers import PositionwiseFeedForward
from layers import PositionalEncoding
from layers import EncoderDecoder
from layers import Encoder
from layers import EncoderLayer
from layers import Decoder
from layers import DecoderLayer
from layers import Embeddings
from layers import Generator

def make_model(src_vocab, tgt_vocab, N = 6, d_model = 512,
               d_ff = 2048, h = 8, dropout = 0.1):
    """
    Helper: Construct a model from hyperparameters.
    """
    c        = copy.deepcopy
    attn     = MultiHeadedAttention(h, d_model)
    ff       = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model    = EncoderDecoder(
        Encoder(                     # Encoder
            EncoderLayer(
                d_model,
                c(attn),
                c(ff),
                dropout
            ),
            N
        ),
        Decoder(                     # Decoder
            DecoderLayer(
                d_model,
                c(attn),
                c(attn),
                c(ff),
                dropout
            ),
            N
        ),
        nn.Sequential(               # Source embedding
            Embeddings(
                d_model,
                src_vocab
            ),
            c(position)
        ),
        nn.Sequential(               # Target embedding
            Embeddings(
                d_model,
                tgt_vocab
            ),
            c(position)
        ),
        Generator(d_model, tgt_vocab) # Generator
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

if __name__ == '__main__':
    model = make_model(10, 10, 2)
    print(model)
