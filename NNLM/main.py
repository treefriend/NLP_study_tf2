import numpy as np
import tensorflow as tf

import models.NNLM as nnlm
import models.config as cfg
import models.data_proc as data

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

    nnlm.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics=['sparse_categorical_crossentropy', 'accuracy'])

    history = nnlm.fit(train_ds, epochs=cfg.epochs, verbose=True)

    '''
    Epoch 1/20
    4069/4069 [==============================] - 79s 19ms/step - loss: 0.9150 - sparse_categorical_crossentropy: 0.9150 - accuracy: 0.9463
    Epoch 2/20
    4069/4069 [==============================] - 79s 19ms/step - loss: 0.6799 - sparse_categorical_crossentropy: 0.6799 - accuracy: 0.9465
    Epoch 3/20
    4069/4069 [==============================] - 81s 20ms/step - loss: 0.6337 - sparse_categorical_crossentropy: 0.6337 - accuracy: 0.9466
    Epoch 4/20
    4069/4069 [==============================] - 79s 19ms/step - loss: 0.5995 - sparse_categorical_crossentropy: 0.5995 - accuracy: 0.9471
    Epoch 5/20
    4069/4069 [==============================] - 79s 19ms/step - loss: 0.5684 - sparse_categorical_crossentropy: 0.5684 - accuracy: 0.9487
    Epoch 6/20
    4069/4069 [==============================] - 79s 20ms/step - loss: 0.5361 - sparse_categorical_crossentropy: 0.5361 - accuracy: 0.9510
    Epoch 7/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.5060 - sparse_categorical_crossentropy: 0.5060 - accuracy: 0.9535
    Epoch 8/20
    4069/4069 [==============================] - 79s 20ms/step - loss: 0.4796 - sparse_categorical_crossentropy: 0.4796 - accuracy: 0.9585
    Epoch 9/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4583 - sparse_categorical_crossentropy: 0.4583 - accuracy: 0.9612
    Epoch 10/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4424 - sparse_categorical_crossentropy: 0.4424 - accuracy: 0.9623
    Epoch 11/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4339 - sparse_categorical_crossentropy: 0.4339 - accuracy: 0.9627
    Epoch 12/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4278 - sparse_categorical_crossentropy: 0.4278 - accuracy: 0.9630
    Epoch 13/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4240 - sparse_categorical_crossentropy: 0.4240 - accuracy: 0.9632
    Epoch 14/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4226 - sparse_categorical_crossentropy: 0.4226 - accuracy: 0.9634
    Epoch 15/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4222 - sparse_categorical_crossentropy: 0.4222 - accuracy: 0.9635
    Epoch 16/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4209 - sparse_categorical_crossentropy: 0.4209 - accuracy: 0.9636
    Epoch 17/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4190 - sparse_categorical_crossentropy: 0.4190 - accuracy: 0.9637
    Epoch 18/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4191 - sparse_categorical_crossentropy: 0.4191 - accuracy: 0.9637
    Epoch 19/20
    4069/4069 [==============================] - 80s 20ms/step - loss: 0.4210 - sparse_categorical_crossentropy: 0.4210 - accuracy: 0.9636
    Epoch 20/20
    4069/4069 [==============================] - 79s 19ms/step - loss: 0.4205 - sparse_categorical_crossentropy: 0.4205 - accuracy: 0.9636
    
    '''