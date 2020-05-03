# -*- coding: utf-8 -*-
#  @Time    : 2020-05-02 12:17
#  @Author  : Shupeng

from collections import defaultdict
from multiprocessing import cpu_count, Pool

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import brown, stopwords

import models.config as cfg

# implemented but not used here
# as words in brown corpus is a word stream but not like sentences
partitions = cpu_count()


def parallize(df, func):
    data_split = np.array_split(df, partitions)
    pool = Pool(partitions)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


class DataProcessor(object):

    def __init__(self, debug=False):
        self.get_dataset()
        if debug:
            self.dataset = brown.words()[:20000]
        else:
            self.dataset = brown.words()
        self.stopwords = stopwords.words('english')
        self.punc = cfg.punctuation
        self.vocab_size = 0
        self.word_index = {}
        self.index_word = {}
        self.word_count = defaultdict(int)
        self.n_gram = cfg.N
        self.gram_dataset = []

    @staticmethod
    def get_dataset():
        nltk.download('brown')
        print('nltk brown loaded')
        nltk.download('stopwords')
        print('nltk stopwords loaded')

    def pre_proc(self):
        lower_ds = [w.lower() for w in self.dataset]
        print('lower done')
        no_punc_ds = [w for w in lower_ds if w not in self.punc]
        print('no punc done')
        no_stopwords = [w for w in no_punc_ds if w not in stopwords.words('english')]
        print('no stopwords done')
        # remove numbers and single letter words
        self.dataset = [w for w in no_stopwords if (w.isalpha() and len(w) > 1)]
        print('isalpha and len filter done')

    def filter_word_count(self):
        word_count = defaultdict(int)
        for w in self.dataset:
            word_count[w] += 1
        self.dataset = [w if word_count[w] < cfg.MIN_COUNT else cfg.MASK_TOKEN for w in self.dataset]

    def calc_data_info(self):
        for w in self.dataset:
            self.word_count[w] += 1
        word_list = [i[0] for i in sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)]
        for idx, word in enumerate(word_list):
            self.index_word[idx] = word
            self.word_index[word] = idx
        self.vocab_size = len(word_list)

    def tokenize_dataset(self):
        self.dataset = [self.word_index[w] for w in self.dataset]

    def convert_gram(self):
        length = len(self.dataset)

        pool = [self.dataset[i:length - self.n_gram + i + 1] for i in range(self.n_gram)]
        self.gram_dataset = np.array(list(map(list, zip(*pool))))


if __name__ == '__main__':
    data_processor = DataProcessor()
    data_processor.pre_proc()
    data_processor.filter_word_count()
    print(len(data_processor.dataset))
