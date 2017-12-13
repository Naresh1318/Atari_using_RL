import tensorflow as tf
import numpy as np
import ops
import os
import sys
from PIL import Image


# TODO: Predicts only greyscale image for now

class Predict_frame:
    def __init__(self):
        self.input_frames = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name='input_frames')
        self.target_frame = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 1], name='target_frame')
        self.action_performed = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='action_performed')
        self.global_step = tf.placeholder(dtype=tf.int32, shape=[], name='Global_step')
        self.n_epochs = 50
        self.generator_lr = 0.005  # TODO: Anneal the learning rate?
        self.discriminator_lr = 0.00005
        self.batch_size = 32
        self.beta1 = 0.5
        self.discriminator_weight = 1.0
        self.l1_weight = 100.0
        self.clip_weight = 10.0
        self.logdir = './Results/prediction_model_skip'
        self.data_dir = '../Dataset/Breakout'
        self.saver_path = './Results/prediction_model_skip/Saved_models'

    def generator(self, x, action, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # TODO: Use a better network for video frame prediction

        x = tf.divide(x, 255.0)
        # Encoder
        conv_1 = ops.lrelu(ops.cnn_2d(x, weight_shape=[4, 4, 4, 64], strides=[1, 2, 2, 1],
                                      padding="SAME", name='g_e_conv_1'))
        conv_2 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_1, weight_shape=[4, 4, 64, 128], strides=[1, 2, 2, 1],
                                                     padding="SAME", name='g_e_conv_2'),
                                          center=True, scale=True, is_training=True, scope='g_e_batch_Norm_2'))
        conv_3 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_2, weight_shape=[4, 4, 128, 256],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='g_e_conv_3'),
                                          center=True, scale=True, is_training=True, scope='g_e_batch_Norm_3'))
        conv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_3, weight_shape=[4, 4, 256, 512],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='g_e_conv_4'),
                                          center=True, scale=True, is_training=True, scope='g_e_batch_Norm_4'))
        conv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_4, weight_shape=[4, 4, 512, 512],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='g_e_conv_5'),
                                          center=True, scale=True, is_training=True, scope='g_e_batch_Norm_5'))
        conv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_5, weight_shape=[4, 4, 512, 512],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='g_e_conv_6'),
                                          center=True, scale=True, is_training=True, scope='g_e_batch_Norm_6'))
        conv_6_reshaped = tf.reshape(conv_6, [-1, 2 * 2 * 512], name='g_conv_6_reshape')

        action_dense_1 = ops.dense(action, 4, 2048, name='g_action_dense_1')
        action_dense_2 = tf.multiply(conv_6_reshaped, action_dense_1, name='g_action_dense_2')

        action_dense_2_reshaped = tf.reshape(action_dense_2, [-1, 2, 2, 512])

        # Decoder
        dconv_1 = ops.lrelu(ops.batch_norm(
            ops.cnn_2d_trans(action_dense_2_reshaped, weight_shape=[2, 2, 512, 512], strides=[1, 2, 2, 1],
                             output_shape=[self.batch_size, action_dense_2_reshaped.get_shape()[1].value * 2-1,
                                           action_dense_2_reshaped.get_shape()[2].value * 2-1,
                                           512], name='g_d_dconv_1'), center=True, scale=True, is_training=True,
            scope='g_d_batch_Norm_1'))
        dconv_1 = tf.concat([dconv_1, conv_5], axis=3)
        dconv_2 = ops.lrelu(ops.batch_norm(
            ops.cnn_2d_trans(dconv_1, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1],
                             output_shape=[self.batch_size, dconv_1.get_shape()[1].value * 2,
                                           dconv_1.get_shape()[2].value * 2, 512], name='g_d_dconv_2'), center=True,
            scale=True, is_training=True, scope='g_d_batch_Norm_2'))
        dconv_2 = tf.concat([dconv_2, conv_4], axis=3)
        dconv_3 = ops.lrelu(ops.batch_norm(
            ops.cnn_2d_trans(dconv_2, weight_shape=[4, 4, 256, 1024], strides=[1, 2, 2, 1],
                             output_shape=[self.batch_size, dconv_2.get_shape()[1].value * 2-1,
                                           dconv_2.get_shape()[2].value * 2-1, 256], name='g_d_dconv_3'), center=True,
            scale=True, is_training=True, scope='g_d_batch_Norm_3'))
        dconv_3 = tf.concat([dconv_3, conv_3], axis=3)
        dconv_4 = ops.lrelu(
            ops.batch_norm(ops.cnn_2d_trans(dconv_3, weight_shape=[4, 4, 128, 512], strides=[1, 2, 2, 1],
                                            output_shape=[self.batch_size, dconv_3.get_shape()[1].value * 2-1,
                                                          dconv_3.get_shape()[2].value * 2-1, 128],
                                            name='g_d_dconv_4'), center=True, scale=True, is_training=True,
                           scope='g_d_batch_Norm_4'))
        dconv_4 = tf.concat([dconv_4, conv_2], axis=3)
        dconv_5 = ops.lrelu(
            ops.batch_norm(ops.cnn_2d_trans(dconv_4, weight_shape=[4, 4, 64, 256], strides=[1, 2, 2, 1],
                                            output_shape=[self.batch_size, dconv_4.get_shape()[1].value * 2,
                                                          dconv_4.get_shape()[2].value * 2, 64],
                                            name='g_d_dconv_5'), center=True, scale=True, is_training=True,
                           scope='g_d_batch_Norm_5'))
        dconv_5 = tf.concat([dconv_5, conv_1], axis=3)
        output = tf.nn.tanh(ops.cnn_2d_trans(dconv_5, weight_shape=[4, 4, 1, 128], strides=[1, 2, 2, 1],
                                             output_shape=[self.batch_size, dconv_5.get_shape()[1].value * 2,
                                                           dconv_5.get_shape()[2].value * 2, 1], name='g_output'))
        return output

    def discriminator(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv_1 = ops.lrelu(ops.batch_norm(ops.cnn_2d(x, weight_shape=[4, 4, 5, 64],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='dis_conv_1'),
                                          center=True, scale=True, is_training=True, scope='dis_batch_Norm_1'))
        conv_2 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_1, weight_shape=[4, 4, 64, 128],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='dis_conv_2'),
                                          center=True, scale=True, is_training=True, scope='dis_batch_Norm_2'))
        conv_3 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_2, weight_shape=[4, 4, 128, 256],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='dis_conv_3'),
                                          center=True, scale=True, is_training=True, scope='dis_batch_Norm_3'))
        conv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_3, weight_shape=[4, 4, 256, 512],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='dis_conv_4'),
                                          center=True, scale=True, is_training=True, scope='dis_batch_Norm_4'))
        conv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_4, weight_shape=[4, 4, 512, 512],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='dis_conv_5'),
                                          center=True, scale=True, is_training=True, scope='dis_batch_Norm_5'))
        conv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_5, weight_shape=[4, 4, 512, 512],
                                                     strides=[1, 2, 2, 1], padding="SAME", name='dis_conv_6'),
                                          center=True, scale=True, is_training=True, scope='dis_batch_Norm_6'))
        conv_7 = tf.reshape(conv_6, [-1, 2 * 2 * 512])
        output = ops.dense(conv_7, 2 * 2 * 512, 1, name='dis_output')
        return output

    def train(self):
        with tf.variable_scope(tf.get_variable_scope()):
            generated_image = self.generator(self.input_frames, self.action_performed)

        discriminator_real_input = tf.concat([self.input_frames, self.target_frame], axis=3)
        discriminator_fake_input = tf.concat([self.input_frames, generated_image], axis=3)

        with tf.variable_scope(tf.get_variable_scope()):
            real_discriminator_op = self.discriminator(discriminator_real_input)
            fake_discriminator_op = self.discriminator(discriminator_fake_input, reuse=True)

        # GAN losses
        generator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                             (labels=tf.ones_like(fake_discriminator_op), logits=fake_discriminator_op))
        discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                 (labels=tf.zeros_like(fake_discriminator_op),
                                                  logits=fake_discriminator_op))
        discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                 (labels=tf.ones_like(real_discriminator_op),
                                                  logits=real_discriminator_op))

        generated_image_clipped = tf.clip_by_value(generated_image, 0, 1)

        clipping_loss = tf.reduce_mean(tf.square(generated_image_clipped - generated_image))

        eps = 1e-5

        target_frame = tf.divide(self.target_frame, 255.0)
        l1_loss = tf.reduce_mean(tf.abs(generated_image - target_frame + eps))

        discriminator_loss = discriminator_fake_loss + discriminator_real_loss

        generator_loss = self.discriminator_weight * generator_fake_loss + \
                         self.l1_weight * l1_loss + self.clip_weight * clipping_loss

        # Collect trainable parameter
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'dis_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        g_learning_rate = tf.train.exponential_decay(self.generator_lr, self.global_step,
                                                     1, 0.999, staircase=True)
        d_learning_rate = tf.train.exponential_decay(self.discriminator_lr, self.global_step,
                                                     1, 0.999, staircase=True)

        generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=self.beta1).minimize(generator_loss,
                                                                                                 var_list=g_vars)
        discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=self.beta1).minimize(discriminator_loss,
                                                                                                     var_list=d_vars)

        tf.summary.scalar(name='l1_loss', tensor=l1_loss)
        tf.summary.scalar(name='discriminator_loss', tensor=discriminator_loss)
        tf.summary.scalar(name='generator_fake_loss', tensor=generator_fake_loss)
        tf.summary.scalar(name='generator_loss', tensor=generator_loss)
        tf.summary.scalar(name='generator_lr', tensor=g_learning_rate)
        tf.summary.scalar(name='discriminator_lr', tensor=d_learning_rate)
        tf.summary.scalar(name='clipping_loss', tensor=clipping_loss)
        tf.summary.image(name='Generated_image', tensor=generated_image_clipped)
        tf.summary.image(name='Target_image', tensor=self.target_frame)

        # TODO: Currently only shows latest input frame
        tf.summary.image(name='Input_frame_0', tensor=tf.reshape(self.input_frames[:, :, :, 0], [-1, 84, 84, 1]))
        tf.summary.image(name='Input_frame_1', tensor=tf.reshape(self.input_frames[:, :, :, 1], [-1, 84, 84, 1]))
        tf.summary.image(name='Input_frame_2', tensor=tf.reshape(self.input_frames[:, :, :, 2], [-1, 84, 84, 1]))
        tf.summary.image(name='Input_frame_3', tensor=tf.reshape(self.input_frames[:, :, :, 3], [-1, 84, 84, 1]))

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        step = 1

        with tf.Session() as sess:
            sess.run(init)

            file_writer = tf.summary.FileWriter(logdir=self.logdir + "/Tensorboard", graph=sess.graph)

            # TODO: Train on 1 step prediction objective later extend
            train_input, train_action, train_target = self.training_data()
            input_size = len(train_action)
            for e in range(self.n_epochs):
                n_batches = int(input_size / self.batch_size)
                for batch in range(n_batches):
                    batch_indx = np.random.permutation(input_size)[:self.batch_size]
                    batch_frame_input = train_input[batch_indx]
                    batch_action_input = train_action[batch_indx]

                    batch_action_input = tf.reshape(tf.one_hot(batch_action_input, depth=4), [self.batch_size, 4])

                    batch_action_input = sess.run(batch_action_input)

                    batch_target = train_target[batch_indx]

                    for i in range(1):
                        sess.run(discriminator_optimizer,
                                 feed_dict={self.input_frames: batch_frame_input, self.target_frame: batch_target,
                                            self.action_performed: batch_action_input,
                                            self.global_step: step})

                    for i in range(1):
                        _, s, l, dl, gl = sess.run([generator_optimizer, summary_op, l1_loss,
                                                    discriminator_loss, generator_fake_loss],
                                                   feed_dict={self.input_frames: batch_frame_input,
                                                              self.target_frame: batch_target,
                                                              self.action_performed: batch_action_input,
                                                              self.global_step: step})

                    file_writer.add_summary(s, global_step=step)

                    print(
                        "\rEpoch: {}/{} \t Batch: {}/{}  l1_loss: {} disc_loss: {} gen_loss: {}".format(e,
                                                                                                        self.n_epochs,
                                                                                                        batch,
                                                                                                        n_batches, l,
                                                                                                        dl,
                                                                                                        gl), end="")
                    sys.stdout.flush()
                    step += 1

                    if step % 500 == 0:
                        saver.save(sess, save_path=self.saver_path + "/model", global_step=step)
                print("\n")
            # Save the final model
            saver.save(sess, save_path=self.saver_path + "/model", global_step=step)

    def training_data(self):
        # TODO: Remove the first 5 to 10 frames from each episode?
        train_input = []
        train_action = []
        train_target = []

        episode_dir = sorted([self.data_dir + "/train/" + p for p in os.listdir(self.data_dir + "/train/")])
        n_episodes = len(episode_dir)
        print("Reading training images!")
        for e_i, episode in enumerate(episode_dir):
            print("Reading training image from episode: {}/{}".format(e_i + 1, n_episodes))
            frames = sorted([f for f in os.listdir(episode) if f.endswith(".png")])
            with open(episode + "/action.txt") as action_file:
                action_log = action_file.read()

            train_action.extend(
                [int(a) for i, a in enumerate(action_log.split("\n")[3:-1])])

            # TODO: Using this for grayscale images only
            for f_indx in range(len(frames)):
                frames_to_use = frames[f_indx:f_indx + 5]
                if len(frames_to_use) < 5:
                    continue
                for i, f in enumerate(frames_to_use):
                    img = ops.convert_to_gray_n_resize(np.array(Image.open(episode + "/" + f)))
                    img = np.expand_dims(img, axis=2)
                    if i == 0:
                        train_frames = img.copy()
                    elif i < 4:
                        train_frames = np.append(train_frames, img, axis=2)
                    else:
                        train_target.append(img)
                train_input.append(train_frames)
        print("Input dataset constructed")
        train_input = np.array(train_input).reshape([-1, 84, 84, 4])  # the last 4 appended frames are useless
        train_action = np.array(train_action).reshape([-1, 1])
        train_target = np.array(train_target).reshape([-1, 84, 84, 1])

        return train_input, train_action, train_target


model = Predict_frame()
model.train()
