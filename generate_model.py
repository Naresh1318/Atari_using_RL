import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import ops
import os
import sys
from PIL import Image

# TODO: Predicts only grayscale image for now


class predict_frame:
    def __int__(self):
        self.input_frames = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name='input_frames')
        self.target_frame = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 1], name='target_frame')
        self.action_performed = tf.placeholder(dtype=tf.int32, shape=[None, 4], name='action_performed')
        self.n_epochs = 100
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.momentum = 0.9
        self.logdir = './Results/prediction_model'
        self.data_dir = '../Dataset/Breakout'
        self.saver_path = './Prediction_model'

    def model(self, x, action, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # TODO: Use a better network for video frame prediction
        # Encoder
        conv_1 = tf.nn.relu(ops.cnn_2d(x, weight_shape=[6, 6, 4, 64], strides=[1, 2, 2, 1], name='conv_1'))
        conv_2 = tf.nn.relu(ops.cnn_2d(conv_1, weight_shape=[6, 6, 64, 64], strides=[1, 2, 2, 1],
                                       name='conv_2', padding="SAME"))
        conv_3 = tf.nn.relu(ops.cnn_2d(conv_2, weight_shape=[6, 6, 64, 64], strides=[1, 2, 2, 1],
                                       name='conv_3', padding="SAME"))
        conv_3_flatten = tf.reshape(conv_3, shape=[-1, 6400], name='reshape_1')
        dense_1 = ops.dense(conv_3_flatten, 6400, 1024, name='dense_1')
        dense_2 = ops.dense(dense_1, 1024, 2048, name='dense_2')
        action_dense_1 = ops.dense(action, 4, 2048, name='action_dense_1')
        dense_2_action = tf.multiply(dense_2, action_dense_1, name='dense_2_action')

        # Decoder
        dense_3 = ops.dense(dense_2_action, 2048, 1024, name='dense_3')
        dense_4 = tf.nn.relu(ops.dense(dense_3, 1024, 6400, name='dense_4'))
        dense_4_reshaped = tf.reshape(dense_4, shape=[-1, 10, 10, 64], name='dense_4_reshaped')
        conv_t_1 = tf.nn.relu(ops.cnn_2d_trans(dense_4_reshaped, weight_shape=[6, 6, 64, 64],
                                               strides=[1, 2, 2, 1], output_shape=[-1, 20, 20, 64], name='conv_t_1'))
        conv_t_2 = tf.nn.relu(ops.cnn_2d_trans(conv_t_1, weight_shape=[6, 6, 64, 64],
                                               strides=[1, 2, 2, 1], output_shape=[-1, 40, 40, 64], name='conv_t_2'))
        output = ops.cnn_2d_trans(conv_t_2, weight_shape=[6, 6, 3, 64],
                                  strides=[1, 2, 2, 1], output_shape=[-1, 84, 84, 1], name='output_image')
        return output

    def train(self):

        with tf.variable_scope("prediction_model"):
            generated_image = self.model(self.input_frames, self.action_performed)

        generated_image_clipped = tf.clip_by_value(generated_image, 0, 1)

        clipping_loss = tf.reduce_mean(tf.square(generated_image_clipped - generated_image))

        eps = 1e-5
        l1_loss = tf.reduce_mean(tf.abs(generated_image - self.target_frame + eps))

        loss = 0.9 * l1_loss + 0.1 * clipping_loss

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(loss)

        tf.summary.scalar(name='l1_loss', tensor=l1_loss)
        tf.summary.scalar(name='clipping_loss', tensor=clipping_loss)
        tf.summary.image(name='Generated_image', tensor=generated_image_clipped)
        tf.summary.image(name='Target_image', tensor=self.target_frame)

        # TODO: Currently only shows latest input frame
        tf.summary.image(name='Input_frame_0', tensor=self.input_frames[:, :, :, 0])
        tf.summary.image(name='Input_frame_1', tensor=self.input_frames[:, :, :, 1])
        tf.summary.image(name='Input_frame_2', tensor=self.input_frames[:, :, :, 2])
        tf.summary.image(name='Input_frame_3', tensor=self.input_frames[:, :, :, 3])

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        step = 0

        with tf.Session() as sess:
            sess.run(init)

            file_writer = tf.summary.FileWriter(logdir=self.logdir, graph=sess.graph)

            # TODO: Train on 1 step prediction objective later extend
            train_input, train_action, train_target = self.training_data()
            input_size = len(train_action)
            for e in range(self.n_epochs):
                n_batches = int(input_size/self.batch_size)
                for batch in range(n_batches):
                    batch_indx = np.random.permutation(input_size)[:self.batch_size]
                    batch_frame_input = train_input[batch_indx]
                    batch_action_input = train_action[batch_indx]

                    batch_action_input = tf.one_hot(batch_action_input, depth=4)

                    batch_target = train_target[batch_indx]

                    _, l = sess.run([optimizer, loss], feed_dict={self.input_frames: batch_frame_input,
                                                   self.action_performed: batch_action_input,
                                                   self.target_frame: batch_target})

                    if step % 1000 == 0:
                        s = sess.run(summary_op, feed_dict={self.input_frames: batch_frame_input,
                                                            self.action_performed: batch_action_input,
                                                            self.target_frame: batch_target})
                        file_writer.add_summary(s)
                    if step % 10000 == 0:
                        saver.save(sess, save_path=self.saver_path)
                    print("Epoch: {}/{} Batch: {}/{} Loss: {}".format(e, self.n_epochs, batch, n_batches, l), end="")
                    sys.stdout.flush()
                    step += 1
                print("\n")




    def training_data(self):
        # TODO: Remove the first 5 to 10 frames from each episode?
        train_input = []
        train_action = []
        train_target = []

        episode_dir = [self.data_dir + "/train/" + p for p in os.listdir(self.data_dir + "/train/")]

        for episode in episode_dir:
            frames = [f for f in os.listdir(episode) if f.endswith(".png")]
            with open(episode_dir+"/action.txt") as action_file:
                action_log = action_log.read()

            train_action.extend([int(a) for a in action_log.split("\n")])

            # TODO: Using this for grayscale images only
            for f_indx in range(len(frames)):
                try:
                    frames_to_use = frames[f_indx:f_indx+4]
                    for i, f in enumerate(frames_to_use):
                        img = ops.convert_to_gray_n_resize(np.array(Image.open(self.data_dir + "/train/" + f)))
                        img = np.expand_dims(img, axis=2)
                        if i == 0:
                            train_frames = img.copy()
                        elif i < 4:
                            train_frames = np.append(train_frames, img, axis=2)
                        else:
                            train_target.append(img)
                except IndexError:
                    print("Input dataset constructed")
                train_input.append(train_frames)
        train_input = np.array(train_input).reshape([-1, 84, 84, 4])
        train_action = np.array(train_action).reshape([-1, 1])
        train_target = np.array(train_target).reshape([-1, 84, 84, 1])

        return train_input, train_action, train_target

model = predict_frame()
