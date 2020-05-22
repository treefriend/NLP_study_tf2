# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 10:26
#  @Author  : Shupeng

import torch.nn as nn

from LayerNorm import LayerNorm
from clone_util import clone


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
