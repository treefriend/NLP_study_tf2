# -*- coding: utf-8 -*-
#  @Time    : 2020-04-26 08:14
#  @Author  : Shupeng

import numpy as np
import tensorflow as tf


class PositionEncoder(object):
    def __init__(self):
        super(PositionEncoder, self).__init__()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def position_encoding(self, pos, d_model):
        angle_rads = self.get_angles(np.arange(pos)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
