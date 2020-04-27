# -*- coding: utf-8 -*-
#  @Time    : 2020-04-26 14:36
#  @Author  : Shupeng

import tensorflow as tf

from Decoder import Decoder
from Encoder import Encoder


class Transformer(tf.keras.models.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, drop_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input, target, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            target, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000
    )

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)
    print('fn_out.shape', fn_out.shape)
