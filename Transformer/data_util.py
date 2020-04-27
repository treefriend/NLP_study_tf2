# -*- coding: utf-8 -*-
#  @Time    : 2020-04-25 18:04
#  @Author  : Shupeng

import tensorflow_datasets as tfds
import tensorflow as tf
from config import *


def load_data():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['train']

    if os.path.exists(pt_encoder_path):
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file(pt_encoder_path)
    else:
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples),
            target_vocab_size=2 ** 13
        )
        tokenizer_pt.save_to_file(pt_encoder_path)

    if os.path.exists(en_encoder_path):
        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_encoder_path)
    else:
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples),
            target_vocab_size=2 ** 13
        )
        tokenizer_en.save_to_file(en_encoder_path)

    return train_examples, val_examples, tokenizer_en, tokenizer_pt


def filter_max_length(x, y, max_length=max_length):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def get_train_dataset(tokenizer_pt, tokenizer_en, train_examples, buffer_size, batch_size):
    def encode(lang1, lang2): # lang1: pt, lang2: en
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def get_train_proc(train_examples, buffer_size):
        return (train_examples
                .map(tf_encode)
                .filter(filter_max_length)
                # cache the dataset to memory, to get a speedup while reading from it
                .cache()
                .shuffle(buffer_size))

    train_preprocessed = get_train_proc(train_examples, buffer_size)
    return (train_preprocessed
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE))


def get_val_dataset(tokenizer_pt, tokenizer_en,val_examples, batch_size):
    # 这里将encode,tf_encode和get_val_proc作为get_val_dataset的内部函数
    # 其实这里遇到了问题，就是在get_train_dataset里面也要写一份
    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def get_val_proc(val_examples):
        return (val_examples
                .map(tf_encode)
                .filter(filter_max_length))

    val_preprocessed = get_val_proc(val_examples)
    return (val_preprocessed
            .padded_batch(batch_size, padded_shapes=([None], [None])))


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


if __name__ == '__main__':
    # load source data by tensorflow.dataset
    train_examples, val_examples, tokenizer_en, tokenizer_pt = load_data()

    # test the tokenizer
    sample_string = 'Transformer is awesome.'

    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    # take a look at the train dataset and test dataset

    train_dataset = get_train_dataset(train_examples, buffer_size, batch_size)
    print(train_dataset.shape)
