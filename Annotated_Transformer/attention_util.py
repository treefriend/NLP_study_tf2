# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 13:22
#  @Author  : Shupeng

import math

import torch
import torch.functional as F


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
    The two most commonly used attention functions are additive attention,
    and dot-product(multiplicative) attention. Here we adopt the dot product one, 
    and applied a scaling factor. Additive attention computes the compatibility
    using a feed-forward network with a single hidden layer. While the two are similar
    in theoretical complexity, dot-product attention is much faster and more space-efficient 
    in practice, since it can be implemented using highly optimized matrix
    multiplication code
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
