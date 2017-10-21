import numpy as np
import tensorflow as tf
import gym
import time
from collections import deque
import mission_control as mc
import ops

# Setup the environment
env = gym.make('CartPole-v0')

# Placeholders
# TODO: make the shape of X_input generatized
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='Observations')
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, env.action_space.n], name='Target_Q_values')


# Dense Fully Connected Agent
def agent(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    dense_1 = ops.dense(x, 8, mc.dense_1, name='dense_1')
    dense_2 = ops.dense(dense_1, mc.dense_1, mc.dense_2, name='dense_2')
    dense_3 = ops.dense(dense_2, mc.dense_2, mc.dense_3, name='dense_3')
    output = ops.dense(dense_3, mc.dense_3, env.action_space.n, name='output')
    return output


def collect_observations(sess):
    pass


def train(train_model=True):
    q_values = agent(X_input)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=q_values, labels=Y_target))

    optimizer = tf.train.AdamOptimizer(learning_rate=mc.learning_rate).minimize(loss)

    # Create the summary for tensorboard
    tf.summary.scalar(name='loss', tensor=loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        observations = collect_observations(sess)
