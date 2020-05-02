# -*- coding: utf-8 -*-
#  @Time    : 2020-05-02 11:56
#  @Author  : Shupeng

N = 6  # n gram
V = 17964  # vocab_size
M = 100  # embedding dim
H = 60  # hidden units

punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
               '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '``', "''", '--']
MIN_COUNT = 3
MASK_TOKEN = '<MASK>'
checkpoint_path = './checkpoints/train'
epochs = 10
batch_size = 100
