# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 12:10
#  @Author  : Shupeng
import numpy as np
import torch


def subsequent_mask(size):
    """
    We also modify the self-attention sub-layer in the decoder stack to prevent positions
    from attending to subsequent positions. This mask, combined with the fact that output
    embeddings are offset by one position, ensure that the predictions for position k can
    depend only on the known outputs at position less than k
    """
    """
    这里注意np.triu这个函数，和np.tril对应，分别用来取一个方阵的上三角和下三角矩阵
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
