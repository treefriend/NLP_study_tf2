# -*- coding: utf-8 -*-
#  @Time    : 2020-05-22 11:07
#  @Author  : Shupeng

import torch.nn as nn

"""
During training, we employed label smoothing of value epsilon_ls=0.1.
This hurts perplexity, as the model learns to be more unsure,
but improves accuracy and BLEU score

We implement label smoothing using the KV div loss. Instead of using a 
one-hot target distribution, we create a distribution that has 
confidence of the correct work and the rest of smoothing mass 
distributed throughout the vocabulary
"""
import torch
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert (x.size(1) == self.size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
