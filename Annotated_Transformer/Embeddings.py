# -*- coding: utf-8 -*-
#  @Time    : 2020-05-21 22:35
#  @Author  : Shupeng

import math

import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        The reason we increase the embedding values before the addition is
        to make the positional encoding relatively smaller. This means the
        original meaning in the embedding vector won’t be lost when we add
        them together.
        """
        return self.lut(x) * math.sqrt(self.d_model)
