# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 11:08
#  @Author  : Shupeng

import torch.nn as nn

from LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        """
        这个真的看着不对劲。。。
        tf2实现的Transformer中，Encoder layer的两个sublayer的操作方式都是：
        1. x1=sublayer(x)
        2. x1=dropout(x1)
        3. x1=x+x1 # residual
        3. layernorm(x1)
        
        和这里的实现的区别在哪里呢？
        区别在于，这里的实现，在第一个sublayer输入前就用了layernorm，即对embedding的输入用了
        tf里面那里是没有用的
        另外一点就是这里输出的最后一层不是layernorm
        tf2的版本是layernorm
        
        如果两个N(0,1)相加，那么分布的结果应该是N(0,sqrt(2))
        所以这二个处理的输出还是不太一样的
        当然，因为LayerNorm是有增益参数的
        所以那个参数可以调节出一个合适的值
        """

        """
        好吧，找到上面问题的解决之道了
        这份代码，在Encoder和Decoder输出的时候，分别单独添加了一个LayerNorm
        也就是这里相当于额外添加了一个输入层的LayerNorm
        这样子确保了第一个x就是N(0,1)的
        关于LayerNorm记得参数貌似只有一份，即共享的
        """
        return x + self.dropout(sublayer(self.norm(x)))
