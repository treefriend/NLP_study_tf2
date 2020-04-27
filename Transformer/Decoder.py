# -*- coding: utf-8 -*-
# @Time    : 2020-04-26 20:13
# @Author  : Shupeng

import tensorflow as tf
from PositionEncoder import PositionEncoder
from DecoderLayer import DecoderLayer
from Encoder import Encoder

class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,
                 max_position_encoding,drop_rate=0.1,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Decoder,self).__init__(trainable, name, dtype, dynamic, **kwargs)

        self.d_model=d_model
        self.num_layers=num_layers

        self.embedding=tf.keras.layers.Embedding(target_vocab_size,d_model)
        self.pos_encoding=PositionEncoder().position_encoding(max_position_encoding,d_model)

        self.dec_layers=[DecoderLayer(d_model,num_heads,dff,drop_rate)
                         for _ in range(num_layers)]

        self.dropout=tf.keras.layers.Dropout(rate=drop_rate)

    def call(self,x,enc_output,training,
             look_ahead_mask,padding_mask):
        seq_len=tf.shape(x)[1]
        attention_weights={}

        x=self.embedding(x)
        x*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x+=self.pos_encoding[:,:seq_len,:]

        x=self.dropout(x,training=training)

        for i in range(self.num_layers):
            x,block1,block2=self.dec_layers[i](x,enc_output,training,
                                               look_ahead_mask,padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)]=block1
            attention_weights['decoder_layer{}_block2'.format(i+1)]=block2

        return x,attention_weights

if __name__ == '__main__':

    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             max_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print('sample_encoder_output.shape',sample_encoder_output.shape)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             max_position_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)