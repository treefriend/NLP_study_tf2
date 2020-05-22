# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 11:56
#  @Author  : Shupeng

import torch.nn as nn

from SublayerConnection import SublayerConnection
from clone_util import clone


class DecoderLayer(nn.Module):
    """Decoder is made of:
    self-attention,
    src-attention(i.e. encoder decoder attention,
    and feed forward
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_fwd = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow figure for connections"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_fwd)
