# -*- coding: utf-8 -*-
#  @Time    : 2020-04-26 12:09
#  @Author  : Shupeng

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout

from MultiHeadAttention import MultiHeadAttention


class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)
        self.dropout3 = Dropout(drop_rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        在Encoder中，self-attention层中的每个位置can attend to all positions in the previous layer of the encoder
        Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions
        in the decoder up to and including that position. We need to prevent leftward information flow in the decoder
        to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out
        (setting to -inf) all values in the input of the softmax which correspond to illegal connections.
        """
        # 对于每一个batch，encoder运行一次，decoder要运行多次
        # decoder的运行次数由target sequence length决定
        # 不断预测一个batch中的下一个词，直到遇到结束符号或者最长length停止预测
        # decoder每次运行，都会有新的look_ahead_mask，mask掉该位置之后的所有位置
        # 这里新的look_ahead_mask指的是上面create_look_ahead_mask函数中新的一行，比如预测第一个词有第一行，第二个次用第二行
        # create_look_ahead_mask创造出的是一个下三角，对角线为1，shape为(seq_len, seq_len)的矩阵
        attn1, attn_weight_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # 这里是直接拿encoder的output作为encoder-decoder-multi-head-attention的query和key
        # 拿decoder的self-attention得到的那个作为value
        # mha的参数顺序为v,k,q,decoder中self-attention的输出是作为query的
        # padding_mask是针对encoder那些为<PAD>的token，而非decoder中的什么
        # 对于encoder中token为<PAD>的位置，要在encoder-decoder-multi-head-attention中将其注意力减少到0（对应的在softmax之前减小
        # 到-inf)，为什么不再softmax之后设置为0呢？因为要保证softmax加和为1
        attn2, attn_weight_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_outupt = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weight_block1, attn_weight_block2


if __name__ == '__main__':
    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512))
    )
