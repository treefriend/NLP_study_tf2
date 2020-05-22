# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 10:14
#  @Author  : Shupeng

import seaborn
import torch.nn as nn

seaborn.set_context(context="talk")


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Based for this and many other models
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        " Take in and process masked src and target sequences"
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
