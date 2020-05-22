# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 10:22
#  @Author  : Shupeng
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    "Define standard linear + softmax generation step"

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
