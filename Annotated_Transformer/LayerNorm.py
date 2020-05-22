# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 10:33
#  @Author  : Shupeng

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """这里感觉有一个问题：
        LayerNorm应该是作用于batch中的一个样本
        x.shape应该是(batch_size，token_length，embedding_dimension)
        如果是作用于一个样本，那么mean和std的dim应该是[-1,-2]
        这里只用-1的话，就是在word或word piece级别做Norm了
        除非x做了flatten，这个可以通过feature怎么初始化来判断
        features=token_length*embedding_dim的话，应该就是flatten了
        不过考虑到这也是一个layer，貌似应该flatten才对？
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
