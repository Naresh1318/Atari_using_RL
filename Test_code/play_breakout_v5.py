import datetime
import itertools
import os
import random
import sys
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf

import ops
from Test_code import mission_control_v3 as mc

# Setup the environment
env = gym.make('BreakoutDeterministic-v4')

# Placeholders
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 2], name='Observations')
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, env.action_space.n], name='Target_Q_values')


# Dense Fully Connected Agent
def get_agent(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    x = tf.divide(x, 255.0, name='Normalize')
    conv_1 = tf.nn.relu(ops.cnn_2d(x, weight_shape=mc.conv_1, strides=mc.stride_1, name='conv_1'))
    conv_2 = tf.nn.relu(ops.cnn_2d(conv_1, weight_shape=mc.conv_2, strides=mc.stride_2, name='conv_2'))
    conv_3 = tf.nn.relu(ops.cnn_2d(conv_2, weight_shape=mc.conv_3, strides=mc.stride_3, name='conv_3'))
    conv_3_r = tf.reshape(conv_3, [-1, 21 * 21 * 64], name='reshape')
    dense_1 = tf.nn.relu(ops.dense(conv_3_r, 21 * 21 * 64, mc.dense_1, name='dense_1'))
    dense_2 = tf.nn.relu(ops.dense(dense_1, mc.dense_1, mc.dense_2, name='dense_2'))
    dense_3 = tf.nn.relu(ops.dense(dense_2, mc.dense_2, mc.dense_3, name='dense_3'))
    dense_4 = tf.nn.relu(ops.dense(dense_3, mc.dense_3, mc.dense_4, name='dense_4'))
    output = ops.dense(dense_4, mc.dense_4, mc.dense_5, name='output')
    return output


def collect_rand_observations(replay_memory):
    print("Collecting Random Observations")
    observation = env.reset()
    observation = ops.convert_to_gray_n_resize(observation)
    observation = np.expand_dims(observation, axis=2)
    last_out = np.zeros_like(observation)
    state = np.append(observation, last_out, axis=2)
    state = np.expand_dims(state, axis=0)
    lives_left = 5
    decay = mc.decay  # Determines the motion tail
    if len(replay_memory) < mc.rand_observation_time:
        for i in range(int(mc.rand_observation_time)):
            action = np.random.randint(low=0, high=env.action_space.n)
            next_state, reward, done, info = env.step(action)
            next_state = ops.convert_to_gray_n_resize(next_state)
            next_state = np.expand_dims(next_state, axis=2)
            next_state = np.expand_dims(next_state, axis=0)

            img_diff = next_state[0, :, :, 0] - state[0, :, :, 0]
            img_motion = np.where(np.abs(img_diff) > 20, 255.0, 0.0)
            output = img_motion + decay * state[0, :, :, 1]
            output = np.expand_dims(output, axis=2)
            output = np.expand_dims(output, axis=0)
            output = np.clip(output, 0.0, 255.0)

            next_states = np.append(next_state, output, axis=3)
            life_lost = 0
            if lives_left - info['ale.lives'] > 0:
                life_lost = 1
                lives_left -= 1
            replay_memory.append((state, action, reward, next_states, done, life_lost))
            state = next_states
            if done:
                lives_left = 5
                observation = env.reset()
                observation = ops.convert_to_gray_n_resize(observation)
                observation = np.expand_dims(observation, axis=2)
                last_out = np.zeros_like(observation)
                state = np.append(observation, last_out, axis=2)
                state = np.expand_dims(state, axis=0)
            print("\rRandom Observation: {}/{}".format(i + 1, mc.rand_observation_time), end="")
            sys.stdout.flush()
    return replay_memory


def make_directories(main_dir):
    main_dir = main_dir + "Time_{}_{}_{}".format(datetime.datetime.now(), mc.n_plays, mc.learning_rate)
    tensorboard_dir = main_dir + "/Tensorboard"
    saved_model_dir = main_dir + "/saved_models"
    log_dir = main_dir + "/logs"
    os.mkdir(main_dir)
    os.mkdir(tensorboard_dir)
    os.mkdir(saved_model_dir)
    os.mkdir(log_dir)
    return tensorboard_dir, saved_model_dir, log_dir


def play(sess, agent, no_plays, log_dir=None, show_ui=False, show_action=False):
    rewards = []
    for p in range(no_plays):
        observation = env.reset()
        observation = ops.convert_to_gray_n_resize(observation)
        observation = np.expand_dims(observation, axis=2)
        state = np.repeat(observation, 4, axis=2)
        state = np.expand_dims(state, axis=0)
        done = False
        reward = 0
        while not done:
            if show_ui:
                env.render()
            if np.random.rand() < 0.05:
                action = np.random.randint(low=0, high=4)
            else:
                action = np.argmax(sess.run(agent, feed_dict={X_input: state}))
            if show_action:
                print(action)
            new_state, r, done, _ = env.step(action)
            r = ops.convert_reward(r)
            next_state = ops.convert_to_gray_n_resize(new_state)
            next_state = np.expand_dims(next_state, axis=2)
            next_state = np.expand_dims(next_state, axis=0)
            # TODO: Interchange next and previous states to check changes
            state = np.append(next_state, state[:, :, :, :3], axis=3)
            reward += r
        rewards.append(reward)
        print("Game: {}/{}".format(p + 1, no_plays))
        print("Reward: {}".format(reward))
        if not log_dir is None:
            with open(log_dir + "/log.txt", "a") as log_file:
                log_file.write("Game: {}/{}\n".format(p + 1, no_plays))
                log_file.write("Reward: {}\n".format(reward))
    print("------------------------------------------------------------------------------------------------------")
    print("Best reward: {}".format(np.amax(rewards)))
    print("Average reward: {}".format(np.mean(rewards)))
    if not log_dir is None:
        with open(log_dir + "/log.txt", "a") as log_file:
            log_file.write("Best reward: {}\n".format(np.amax(rewards)))
            log_file.write("Average reward: {}\n".format(np.mean(rewards)))


def train(train_model=True):
    agent = get_agent(X_input)

    squared_error = tf.square(agent - Y_target)
    sum_squared_error = tf.reduce_sum(squared_error, axis=1)
    loss = tf.reduce_mean(sum_squared_error)

    # TODO: Add loss decay operation
    optimizer = tf.train.RMSPropOptimizer(learning_rate=mc.learning_rate).minimize(loss)

    # Create the summary for tensorboard
    tf.summary.scalar(name='loss', tensor=loss)
    tf.summary.scalar(name='max_q_value', tensor=tf.reduce_max(agent))
    tf.summary.histogram(name='q_values_hist', values=agent)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        if train_model:
            print("Training agent!")
            print("Preparing required directories")
            tensorboard_dir, saved_model_dir, log_dir = make_directories(mc.logdir)

            print("Tensorboard files stores in: {}".format(tensorboard_dir))
            print("Saved models stored in: {}".format(saved_model_dir))
            print("Log files stores in: {}".format(log_dir))

            # Initialize global variables
            sess.run(init)

            # File writer for tensorboard
            writer = tf.summary.FileWriter(logdir=tensorboard_dir, graph=sess.graph)

            # Used to measure time taken
            t1 = time.time()

            # Kinda like the global step, but is not a "Tensor"
            step = 0

            # Get the initial epsilon
            prob_rand = mc.prob_random

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=prob_rand, tag="epsilon")
            writer.add_summary(episode_summary, step)

            # Replay memory
            replay_memory = deque()
            replay_memory = collect_rand_observations(replay_memory)  # Get the initial 50k random observations

            # Save current model parameters
            with open("mission_control_v3.py", "r") as mc_file:
                mission_control_file = mc_file.read()
                with open(log_dir + "/mission_control.txt", "w") as mc_writer:
                    mc_writer.write(mission_control_file)

            for e in range(mc.n_plays):
                with open(log_dir + "/log.txt", "a") as log_file:
                    log_file.write("--------------------------Play: {}/{}------------------------------\n"
                                   .format(e + 1, mc.n_plays))

                # Prepare first observation
                observation = env.reset()
                observation = ops.convert_to_gray_n_resize(observation)
                observation = np.expand_dims(observation, axis=2)
                last_out = np.zeros_like(observation)
                state = np.append(observation, last_out, axis=2)
                state = np.expand_dims(state, axis=0)
                decay = mc.decay  # Determines the motion tail

                episode_rewards = []

                # TODO: Only for breakout
                lives_left = 5
                log_q_values = []
                for t in itertools.count():
                    mini_batch = random.sample(replay_memory, mc.batch_size)

                    agent_input = []
                    agent_target = []
                    for s in range(len(mini_batch)):
                        state_ = mini_batch[s][0]
                        action_ = mini_batch[s][1]
                        reward_ = mini_batch[s][2]
                        next_state_ = mini_batch[s][3]
                        done_ = mini_batch[s][4]
                        lives = mini_batch[s][5]

                        agent_input.append(state_[0])
                        target = sess.run(agent, feed_dict={X_input: state_})
                        if done_ or lives == 0:
                            target[0][action_] = reward_
                            agent_target.append(target[0])
                        else:
                            agent_output = sess.run(agent, feed_dict={X_input: next_state_})
                            target[0][action_] = reward_ + mc.gamma * (np.amax(agent_output))
                            agent_target.append(target[0])

                    # Training the agent for 1 iterations. Finally!!
                    for i in range(mc.fit_epochs):
                        _, l, summary = sess.run([optimizer, loss, summary_op],
                                                 feed_dict={X_input: agent_input, Y_target: agent_target})

                    writer.add_summary(summary, global_step=step)

                    print("\rStep: {} ({}), Play: {}/{}, Loss: {}".format(t, step, e + 1, mc.n_plays, l), end="")
                    sys.stdout.flush()

                    # Collect the next observation
                    if np.random.rand() < prob_rand:
                        action = random.randrange(env.action_space.n)
                    else:
                        q_prediction = sess.run(agent, feed_dict={X_input: state})
                        action = np.argmax(q_prediction)
                        log_q_values.extend(q_prediction)
                    next_state, reward, done, info = env.step(action)
                    next_state = ops.convert_to_gray_n_resize(next_state)
                    next_state = np.expand_dims(next_state, axis=2)
                    next_state = np.expand_dims(next_state, axis=0)

                    img_diff = next_state[0, :, :, 0] - state[0, :, :, 0]
                    img_motion = np.where(np.abs(img_diff) > 20, 255.0, 0.0)
                    output = img_motion + decay * state[0, :, :, 1]
                    output = np.expand_dims(output, axis=2)
                    output = np.expand_dims(output, axis=0)
                    output = np.clip(output, 0.0, 255.0)

                    next_states = np.append(next_state, output, axis=3)

                    life_lost = 0
                    if lives_left - info['ale.lives'] > 0:
                        life_lost = 1
                        lives_left -= 1

                    # Remove old samples from replay memory if it's full
                    if len(replay_memory) > mc.observation_time:
                        replay_memory.popleft()

                    replay_memory.append((state, action, reward, next_states, done, life_lost))
                    state = next_states
                    episode_rewards.append(reward)
                    step += 1

                    prob_rand = ops.anneal_epsilon(prob_rand, step)

                    if done:
                        break
                with open(log_dir + "/log.txt", "a") as log_file:
                    log_file.write("Step: {} ({}), Play: {}/{}, Loss: {}\n".format(t, step, e + 1, mc.n_plays, l))
                    log_file.write("Reward Obtained: {}\n".format(np.sum(episode_rewards)))
                    if log_q_values != []:
                        log_file.write("Average Q Value: {}\n".format(np.mean(log_q_values)))
                    else:
                        log_file.write("All of the actions were random\n")

                print("\nReward Obtained: {}".format(np.sum(episode_rewards)))

                if log_q_values != []:
                    print("Average Q Value: {}".format(np.mean(log_q_values)))
                else:
                    print("All of the actions were random")

                # Save the agent
                saved_path = saver.save(sess, saved_model_dir + '/model', global_step=step)

            print("Time taken of {} Plays on your potato: {:.4f}s".format(mc.n_plays, time.time() - t1))
            print("Average time for each Play: {:.4f}s".format((time.time() - t1) / mc.n_plays))
            print("Tensorboard files saved in: {}".format(tensorboard_dir))
            print("Model saved in: {}".format(saved_path))
            print("Model parameters stored in: {}".format(log_dir + "mission_control.txt"))
            print("Agent get to roll!")
            with open(log_dir + "/log.txt", "a") as log_file:
                log_file.write("Time taken of {} epochs on your potato: {:.4f}s\n".format(mc.n_plays, time.time() - t1))
                log_file.write("Average time for each epoch: {:.4f}s\n".format((time.time() - t1) / mc.n_plays))
        else:
            # Get the latest trained model
            saved_models = os.listdir(mc.logdir)
            latest_saved_model = sorted(saved_models)[-1]
            saver.restore(sess, tf.train.latest_checkpoint(mc.logdir + latest_saved_model + "/saved_models/"))
            print("Getting model from: {}".format(mc.logdir + latest_saved_model + "/saved_models/"))
            print("------------------------Playing----------------------------")
            play(sess=sess, agent=agent, no_plays=mc.n_plays, log_dir=None,
                 show_ui=mc.show_ui, show_action=mc.show_action)


if __name__ == '__main__':
    train(train_model=mc.train_model)
