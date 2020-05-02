# -*- coding: utf-8 -*-
#  @Time    : 2020-05-02 11:55
#  @Author  : Shupeng

import numpy as np
import tensorflow as tf


class NNLM(tf.keras.models.Model):
    def __init__(self, N, V, M, H, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(NNLM, self).__init__(trainable, name, dtype, dynamic, **kwargs)

        self.n_gram = N
        self.vocab_size = V
        self.embedding_dim = M
        self.hidden_unit = H

        # self.input_layer=tf.keras.layers.Input(self.vcoab_size)
        self.embedding_layer = tf.keras.layers.Dense(self.embedding_dim, use_bias=False, activation=None)
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_unit, activation='tanh')
        self.outupt_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, **kwargs):
        x = input(inputs)
        x_embs = [self.embedding_layer(x) for _ in range(self.n_gram)]
        x_total = tf.keras.layers.concatenate(x)
        h = self.hidden_layer(x_total)
        fc = self.output_layer(h)
        return np.argmax(fc)
