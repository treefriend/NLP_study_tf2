# -*- coding: utf-8 -*-
#  @Time    : 2020-05-20 10:25
#  @Author  : Shupeng
import copy

import torch.nn as nn


def clone(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
