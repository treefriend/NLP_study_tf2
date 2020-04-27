# -*- coding: utf-8 -*-
# @Time    : 2020-04-26 09:04
# @Author  : Shupeng

import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense

class MultiHeadAttention(Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads=num_heads
        self.d_model=d_model

        assert self.d_model % self.num_heads==0

        self.depth=self.d_model//self.num_heads

        self.wq=Dense(d_model)
        self.wk=Dense(d_model)
        self.wv=Dense(d_model)

        self.dense=Dense(d_model)

        self.verbose=True

    def split_heads(self,x,batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        :param x:
        :param batch_size:
        :return:
        """
        x=tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3])

    def call(self,v,k,q,mask):
        batch_size=tf.shape(q)[0]
        q=self.wq(q) # (batch_size, seq_len, d_model)
        k=self.wk(k) # (batch_size, seq_len, d_model)
        v=self.wv(v) # (batch_size, seq_len, d_model)

        q=self.split_heads(q,batch_size) # (batch_size, num_heads, seq_len, depth)
        k=self.split_heads(k,batch_size) # (batch_size, num_heads, seq_len, depth)
        v=self.split_heads(v,batch_size) # (batch_size, num_heads, seq_len, depth)

        scaled_attention, attention_weights=self.scaled_dot_product_attention(
            q,k,v,mask
        )

        scaled_attention=tf.transpose(scaled_attention,perm=[0,2,1,3])

        concat_attention=tf.reshape(scaled_attention,
                                    (batch_size,-1,self.d_model))

        output=self.dense(concat_attention)

        return output,attention_weights

    def scaled_dot_product_attention(self,q,k,v,mask):
        '''
        q, k, v mst have matching leading dimensions
        :param q: query shape == (..., seq_len_q, depth)
        :param k: key shape == (..., seq_len_k, depth)
        :param v: value shape == (..., seq_len_v, depth)
        :param mask: Float tensor with shape broadcastable
                     to (..., seq_len_q, seq_len_k). Defaults to None.
        :return:
            output, attention_weights
        '''
        # 这里是一个高维矩阵乘法，因为前两个维度一致，所以可以运行，高维乘法就是最后两个维度按照二维矩阵相乘来进行运算
        # 因为前面的维度都当做batch，所以这里可以保证8个head里面每个head的q和且只和该head的key进行矩阵乘法，即点积
        # 但是因为矩阵乘法要满足左矩阵列数等于右矩阵行数，所以这里有transpose_b=True
        # 对于当做batch_dim的那些维度来说，transpose不影响
        # transpose_b=True只是将(batch_dims,dim_row,dim_col)变成(batch_dims,dim_col,dim_row)
        matmul_qk=tf.matmul(q,k,transpose_b=True)
        # self attn: (batch_size, num_heads, seq_len, seq_len)
        # encoder decoder mha: (batch_size, num_heads, seq_len_decoder, seq_len_encoder)


        dk=tf.cast(tf.shape(k)[-1],tf.float32)
        scaled_attention_logits=matmul_qk/tf.math.sqrt(dk)

        # 这里对mask的处理是加上-inf，这样在softmax之后就是一个逼近0的值
        if mask is not None:
            scaled_attention_logits+=(mask*-1e9)

        attention_weights=tf.nn.softmax(scaled_attention_logits,axis=-1) # (batch_size, num_heads, seq_len, seq_len)

        output=tf.matmul(attention_weights,v) # (batch_size, num_heads, seq_len, depth)

        # 这里有个小问题，就是输出并没有进行复合，论文里是复合了的，从而没有num_heads这个dim了
        return output,attention_weights


if __name__=='__main__':
    temp_mha=MultiHeadAttention(d_model=512,num_heads=8)
    y=tf.random.uniform((1,60,512)) # batch_size, encoder_sequence, d_model
    out,attn=temp_mha(y,y,y,mask=None)
    print('out.shape',out.shape)
    print('attn.shape',attn.shape)