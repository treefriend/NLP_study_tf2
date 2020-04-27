# -*- coding: utf-8 -*-
# @Time    : 2020-04-27 08:39
# @Author  : Shupeng

import tensorflow as tf
from Transformer import Transformer
from data_util import load_data,create_look_ahead_mask,create_padding_mask,get_train_dataset,get_val_dataset
from config import *
# load dataset
train_examples, val_examples, tokenizer_en, tokenizer_pt = load_data()
train_dataset= get_train_dataset(tokenizer_pt, tokenizer_en, train_examples, buffer_size, batch_size)
val_dataset=get_val_dataset(tokenizer_pt, tokenizer_en,val_examples, batch_size)

# set hyper parameters
num_layers=4
d_model=128
dff=512 # relu层的神经元数量
num_heads=8 # depth=128/8=16

input_vocab_size=tokenizer_pt.vocab_size+2
target_vocab_size=tokenizer_en.vocab_size+2
dropout_rate=0.1



# optimizer,优化器就是根据数据表现调整学习率
# 通过继承tf.keras.optimizers.schedules.LearningRateSchedule创建一个可序列化的学习率衰减（也可能增加）时间表
# 优化器可以方便的使用这个类的实例改变学习率，optimizer的核心函数就一个：根据学习率apply_gradient
# 训练的过程就是
# 1. call model
# 2. 计算loss
# 3. 计算gradient
# 4. apply gradient更新model
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model,warmup_steps=4000):
        super(CustomSchedule,self).__init__()

        self.d_model=d_model
        self.d_model=tf.cast(self.d_model,tf.float32)

        self.warmup_steps=warmup_steps

    def __call__(self, step):
        # tf.math.rsqrt(x)=1/tf.math.sqrt(x)
        arg1=tf.math.rsqrt(step)
        arg2=step*(self.warmup_steps**-1.5)
        # 当step小于warmup_steps的时候，min是arg2
        # 当step大于warmup_steps的时候，min是arg1
        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)

learning_rate=CustomSchedule(d_model)
optimizer=tf.keras.optimizers.Adam(learning_rate,
                                   beta_1=0.9,
                                   beta_2=0.98,
                                   epsilon=1e-9)

# loss函数和监测指标
loss_object=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,reduction='none'
)

def loss_function(real, pred):
    # 原来label为0的位置mask掉
    mask=tf.math.logical_not(tf.math.equal(real,0))
    loss_=loss_object(real,pred)

    mask=tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask
    # 计算平均loss，分摊到每个单词的预测
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss=tf.keras.metrics.Mean(name='train_loss')
train_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(
    name='train_accuracy'
)

# train and checkpointing

# create model
transformer=Transformer(num_layers,d_model,num_heads,dff,
                        input_vocab_size,target_vocab_size,
                        pe_input=input_vocab_size,
                        pe_target=target_vocab_size,
                        drop_rate=dropout_rate)

# create the masks
def create_masks(inp,tar):
    # Encoder padding mask
    enc_padding_mask=create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask=create_padding_mask(inp)

    # Used in the 1st attention block in the decoder
    # It is used to pad and mask future tokens in the input received by the decoder
    look_ahead_mask=create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask=create_padding_mask(tar)
    combined_mask=tf.maximum(dec_target_padding_mask,look_ahead_mask)

    return enc_padding_mask,combined_mask,dec_padding_mask

# checkpoint handler
checkpoint_path='./checkpoints/train'
ckpt=tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
ckpt_manager=tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")

EPOCHS=20

'''
The @tf.function trace-compiles train_step into a TF graph for faster
execution. The function specializes to the precise shape of the argument
tensors. To avoid re-tracing due to the variable sequence lengths or variable
batch sizes (the last batch is smaller), sue input signature to specify 
more generic shapes
'''

train_step_signature=[
    tf.TensorSpec(shape=(None,None),dtype=tf.int64),
    tf.TensorSpec(shape=(None,None),dtype=tf.int64)
]

@tf.function(input_signature=train_step_signature)
def train_step(inp,tar):
    tar_inp=tar[:,:-1]
    tar_real=tar[:,1:]

    enc_padding_mask,combined_mask,dec_padding_mask=create_masks(inp,tar_inp)

    with tf.GradientTape() as tape:
        predictions,_=transformer(inp,tar_inp,
                                  True,
                                  enc_padding_mask,
                                  combined_mask,
                                  dec_padding_mask)
        loss=loss_function(tar_real,predictions)

    gradients=tape.gradient(loss,transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real,predictions)

import time

for epoch in range(EPOCHS):
    start=time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp --> portuguese, tar --> english
    for (batch,(inp,tar)) in enumerate(train_dataset):
        train_step(inp,tar)

        if batch % 50==0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch+1,batch,train_loss.result(),train_accuracy.result()
            ))

    if (epoch+1)%5==0:
        ckpt_save_path=ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(
            epoch+1,ckpt_save_path
        ))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch+1,train_loss.result(),train_accuracy.result()
    ))

    print('Time taken for 1 epoch: {} mins\n'.format(
        (time.time()-start)/60
    ))

# Evaluate
def evaluate(inp_sentence):
    start_token=[tokenizer_pt.vocab_size]
    end_token=[tokenizer_pt.vocab_size+1]

    # inp sentence is portuguese, hense adding the start and end token
    inp_sentence=start_token+tokenizer_pt.encode(inp_sentence)+end_token
    encoder_input=tf.expand_dims(inp_sentence,0)

    # as the target is english, the first word to the transformer should be the english start token
    decoder_input=[tokenizer_en.vocab_size]
    output=tf.expand_dims(decoder_input,0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask=create_masks(encoder_input,output)

        # predictions.shape==(batch_size, seq_len, vocab_size)
        predictions, attention_weights=transformer(encoder_input,
                                                   output,
                                                   False,
                                                   enc_padding_mask,
                                                   combined_mask,
                                                   dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions=predictions[:,-1:,:] # (batch_size, 1, vocab_size)

        predicted_id=tf.cast(tf.argmax(predictions,axis=-1),tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id==tokenizer_en.vocab_size+1:
            return tf.squeeze(output,axis=0),attention_weights

        # concatentate the predicted_id to the output which is given to the encoder as its input
        output=tf.concat([output,predicted_id],axis=-1)

    return tf.squeeze(output,axis=0),attention_weights
