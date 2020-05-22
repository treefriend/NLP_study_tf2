# -*- coding: utf-8 -*-
#  @Time    : 2020-05-21 17:25
#  @Author  : Shupeng

import torch.nn as nn

from attention_util import attention
from clone_util import clone


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    With a single attention head, averaging inhibits(抑制，禁止) this.
    """

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equal to d_k
        self.d_k = d_model // h  # 按原文,d_k=512/8=64
        # 应该可以认为第一层embedding的维度就是512，而且还记得positional encoding的长度也是512吗？哈哈
        self.h = h
        # 注意linear输入和输出的dim都是d_model，也就是原文里的512
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # linear layer实现的函数就是wx+b，要求输入的样本每一个都是list才行，不能再嵌套了
        # 其实我觉的torch里面这个view函数很灵性
        # 说白了，在内存堆上面，数据都是线性存储的
        # 而矩阵只是在线性存储上的一个逻辑组织，给每个值添加新的index
        # 每个head的query, key, value三个变换矩阵都是(d_k, d_k)的
        # 这里结合矩阵的技巧，通过一次矩阵乘法实现了8个head一起出来
        # 为什么用8个64的attention，而不是一个512的？
        # 这应该也有一些ensemble的思想在里面了
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final layer
        # 我们的输出是一个512的哦！
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
