import tensorflow as tf
import numpy as np
import scipy.ndimage


def dense(x, n1, n2, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[n1, n2], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=2e-2))
        bias = tf.get_variable('bias', shape=[n2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=2e-2))
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output


def cnn_2d(x, weight_shape, strides, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=2e-2))
        bias = tf.get_variable('bias', shape=[weight_shape[-1]], initializer=tf.truncated_normal_initializer(mean=0, stddev=2e-2))
        output = tf.nn.conv2d(x, filter=weights, strides=strides, padding="VALID", name="Output") + bias
        return output


def convert_to_gray_n_resize(im):
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
    img = scipy.misc.imresize(img_gray, size=[84, 84], interp='bicubic')
    return img


def convert_reward(reward):
    return np.sign(reward)


def anneal_epsilon(epi, step):
    if step < 1e6:
        epi = epi * (.1**1e-6)
    else:
        epi = 0.1
    return epi
