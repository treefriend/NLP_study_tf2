import numpy as np
import tensorflow as tf

import models.NNLM as nnlm
import models.config as cfg
import models.data_proc as data
import time

# load dataset

if __name__ == '__main__':
    dp = data.DataProcessor(debug=False)
    dp.pre_proc()
    dp.filter_word_count()
    dp.calc_data_info()
    dp.tokenize_dataset()
    dp.convert_gram()
    print(len(dp.dataset))
    dataset = tf.cast(dp.gram_dataset, tf.int32)
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

    train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(cfg.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(cfg.batch_size)

    # create model
    nnlm = nnlm.NNLM(N=cfg.N, V=vocab_size, M=cfg.M, H=cfg.H)

    # create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    # create loss function
    # 这里之前用的CategoricalCrossentropy，搞错了
    # 看源码可以直到CategoricalCrossentropy是对应one-hot类型的标签的
    # SparseCategoricalCrossentropy是对应分类编号的标签的
    # 还是要看源码，tf的代码写的好哇，特别是注释
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, word2index[cfg.MASK_TOKEN]))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


    # checkpoint
    @tf.function()
    def train_step(input, target):
        with tf.GradientTape() as tape:
            predictions = nnlm(input)
            loss = loss_function(target, predictions)

        gradients = tape.gradient(loss, nnlm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, nnlm.trainable_variables))
        print('loss', loss)
        train_loss(loss)
        train_accuracy(target, predictions)


    for (batch, (input, target)) in enumerate(test_ds):
        print(batch)
        pred = nnlm(input)
        print('pred for batch {}'.format(batch))
        print(pred)

    for epoch in range(cfg.epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (input, target)) in enumerate(train_ds):
            train_step(input, target)

            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()
            ))

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #
    # nnlm.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # history = nnlm.fit(train_ds, epochs=cfg.epochs, callbacks=early_stopping,
    #                    validation_data=test_ds, verbose=True)
