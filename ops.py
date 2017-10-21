import tensorflow as tf


# Dense
def dense(x, n1, n2, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[n1, n2], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer)
        bias = tf.get_variable('bias', shape=[n2], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer)
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output