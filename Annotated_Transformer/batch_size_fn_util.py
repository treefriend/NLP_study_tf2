# -*- coding: utf-8 -*-
#  @Time    : 2020-05-22 10:56
#  @Author  : Shupeng

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep argumenting batch and calculate total number of tokens + padding"
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
