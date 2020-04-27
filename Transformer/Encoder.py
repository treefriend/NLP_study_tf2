# -*- coding: utf-8 -*-
#  @Time    : 2020-04-26 16:44
#  @Author  : Shupeng

import tensorflow as tf

from EncoderLayer import EncoderLayer
from PositionEncoder import PositionEncoder


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 max_position_encoding, drop_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionEncoder().position_encoding(max_position_encoding,
                                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, drop_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


if __name__ == '__main__':
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             max_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print(sample_encoder_output.shape)
