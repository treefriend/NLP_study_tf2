# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 11:36
#  @Author  : Shupeng

import torch.nn as nn

from SublayerConnection import SublayerConnection
from clone_util import clone


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed-forward"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_fwd = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow figure for connections"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_fwd)
