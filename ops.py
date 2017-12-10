import numpy as np
import scipy.ndimage
import tensorflow as tf


def dense(x, n1, n2, name):
    """
    Dense fully connected layer
    :param x: Tensor, input Tensor
    :param n1: int, float, number of input neurons
    :param n2: int, float, number of output neurons
    :param name: String, Name of the layer on Tensorboard also prefixes this value for the variable name
    :return: Tensor, output tensor after passing it through the dense fully layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[n1, n2], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        bias = tf.get_variable('bias', shape=[n2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output


def cnn_2d(x, weight_shape, strides, name, padding="VALID"):
    """
    A 2d Convolutional layer
    :param x: Tensor, input tensor
    :param weight_shape: Tensor, [filter_size, filter_size, n_input_features, n_output_features]
    :param strides: Tensor, [1, stride_x, stride_y, 1]
    :param name: String, Name of the layer on Tensorboard also prefixes this value for the variable name
    :param padding: String, "VALID" or "SAME"
    :return: Tensor, output tensor after passing it through the convolutional layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        bias = tf.get_variable('bias', shape=[weight_shape[-1]], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        output = tf.nn.conv2d(x, filter=weights, strides=strides, padding=padding, name="Output") + bias
        return output


def cnn_2d_trans(x, weight_shape, strides, output_shape, name):
    """
    Performs convolution transpose
    :param x: Tensor, input tensor
    :param weight_shape: Tensor, [filter_size, filter_size, n_input_features, n_output_features]
    :param strides: Tensor, [1, stride_x, stride_y, 1]
    :param output_shape: List, required output shape
    :param name: String, Name of the layer on Tensorboard also prefixes this value for the variable name
    :return: Tensor, output tensor after passing it through the convolutional transpose layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=weight_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        bias = tf.get_variable('bias', shape=[weight_shape[-2]],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        output = tf.nn.conv2d_transpose(x, weights, output_shape, strides=strides, padding="SAME") + bias
        return output


def convert_to_gray_n_resize(im):
    """
    Converts the input image to gray scale and resize it to 84 x 84
    :param im: 3d image, image to convert
    :return: 2d image matProcessed image
    """
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
    img_gray = np.array(img_gray).astype(np.uint8)
    img = scipy.misc.imresize(img_gray, size=[84, 84], interp='bicubic')
    return np.array(img, dtype=np.uint8)


def convert_reward(reward):
    """
    Convert all rewards obtained to either a 1, 0 or -1. I have no idea why this is done.
    :param reward: float or list, reward/s obtained obtained after each step
    :return: float or list, sign of the rewards obtained
    """
    return np.sign(reward)

