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

        #self.input_layer = tf.keras.layers.Input(batch =self.vocab_size)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size,output_dim=self.embedding_dim)
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_unit, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, x, **kwargs):
        #x = self.input_layer(inputs)
        x = self.embedding_layer(x)
        x=tf.keras.layers.Flatten()(x)
        h = self.hidden_layer(x)
        fc = self.output_layer(h)
        return fc
