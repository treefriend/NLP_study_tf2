# -*- coding: utf-8 -*-
#  @Time    : 2020-05-22 10:23
#  @Author  : Shupeng

import copy

import torch.nn as nn

from Decoder import Decoder
from DecoderLayer import DecoderLayer
from Embeddings import Embeddings
from Encoder import Encoder
from EncoderDecoder import EncoderDecoder
from EncoderLayer import EncoderLayer
from Generator import Generator
from MultiHeadedAttention import MultiHeadedAttention
from PositionEncoding import PositionalEncoding
from PositionWiseFeedForward import PositionWiseFeedForward


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab, c(position))),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
