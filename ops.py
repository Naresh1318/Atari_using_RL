import tensorflow as tf
import cv2


def dense(x, n1, n2, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[n1, n2], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        bias = tf.get_variable('bias', shape=[n2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer)
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output


def cnn_2d(x, weight_shape, strides, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        bias = tf.get_variable('bias', shape=[weight_shape[-1]], initializer=tf.constant_initializer(0.0))
        output = tf.nn.conv2d(x, filter=weights, strides=strides, padding="VALID", name="Output") + bias
        return output


def convert_to_gray_n_resize(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = cv2.resize(im, (84, 84))
    return im
