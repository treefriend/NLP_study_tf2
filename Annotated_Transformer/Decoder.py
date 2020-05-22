# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 11:40
#  @Author  : Shupeng

import torch.nn as nn

from LayerNorm import LayerNorm
from clone_util import clone


class Decoder(nn.Module):
    "Generic N layer decoder with masking"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
