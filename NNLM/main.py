import numpy as np
import tensorflow as tf

import models.NNLM as nnlm
import models.config as cfg
import models.data_proc as data

# load dataset

if __name__ == '__main__':
    dp = data.DataProcessor(debug=True)
    dp.pre_proc()
    dp.filter_word_count()
    dp.calc_data_info()
    dp.tokenize_dataset()
    dp.convert_gram()
    print(len(dp.dataset))
    dataset = dp.gram_dataset
    word2index = dp.word_index
    index2word = dp.index_word
    vocab_size = dp.vocab_size

    # train_test split
    flag = int(len(dataset) * 0.8)
    train_dataset = dataset[:flag]
    test_dataset = dataset[flag:]

    train_X, train_y = np.array(train_dataset[:, :cfg.N - 1]), np.array(train_dataset[:, -1])
    test_X, test_y = np.array(test_dataset[:, :cfg.N - 1]), np.array(test_dataset[:, -1])
    print(test_X[0], test_y[0])

    # create model
    nnlm = nnlm.NNLM(N=cfg.N, V=vocab_size, M=cfg.M, H=cfg.H)

    # # create optimizer
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
    #                                      beta_1=0.9,
    #                                      beta_2=0.98,
    #                                      epsilon=1e-9)
    #
    # # create loss function
    # loss_ojbect = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='None'
    # )
    #
    #
    # def loss_function(real, pred):
    #     mask = tf.math.logical_not(tf.math.equal(real, word2index[cfg.MASK_TOKEN]))
    #     loss_ = loss_ojbect(real, pred)
    #     mask = tf.cast(mask, dtype=loss_.dtype)
    #     loss_ *= mask
    #     return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    #
    #
    # checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    nnlm.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    history = nnlm.fit(x=train_X, y=train_y, batch_size=cfg.batch_size, epochs=cfg.epochs, callbacks=early_stopping,
                       validation_data=(test_X, test_y), verbose=True)
