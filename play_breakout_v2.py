import numpy as np
import tensorflow as tf
import gym
import time
import os
import datetime
import random
from collections import deque
import mission_control as mc
import ops

# Setup the environment
env = gym.make('BreakoutDeterministic-v4')

# Placeholders
# TODO: make the shape of X_input generalized
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name='Observations')
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, env.action_space.n], name='Target_Q_values')


# Dense Fully Connected Agent
# TODO: x -> [210, 160, 4], try this later
def get_agent(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    x = tf.divide(x, 255.0, name='Normalize')
    conv_1 = tf.nn.relu(ops.cnn_2d(x, weight_shape=mc.conv_1, strides=mc.stride_1, name='conv_1'))
    conv_2 = tf.nn.relu(ops.cnn_2d(conv_1, weight_shape=mc.conv_2, strides=mc.stride_2, name='conv_2'))
    conv_3 = tf.nn.relu(ops.cnn_2d(conv_2, weight_shape=mc.conv_3, strides=mc.stride_3, name='conv_3'))
    conv_3_r = tf.reshape(conv_3, [-1, 7*7*64], name='reshape')
    dense_1 = tf.nn.relu(ops.dense(conv_3_r, 7*7*64, mc.dense_1, name='dense_1'))
    output = ops.dense(dense_1, mc.dense_1, mc.dense_2, name='dense_2')
    return output


def collect_rand_observations(replay_memory):
    print("Collecting Random Observations!!")
    observation = env.reset()
    observation = ops.convert_to_gray_n_resize(observation)
    observation = np.expand_dims(observation, axis=2)
    state = np.repeat(observation, 4, axis=2)
    state = np.expand_dims(state, axis=0)
    r = 0
    if len(replay_memory) < mc.rand_observation_time:
        for i in range(int(mc.rand_observation_time)):
            action = np.random.randint(low=0, high=env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            # reward = ops.convert_reward(reward)
            if reward == 1.0:
                r += 1
                print("Here!!")
            next_state = ops.convert_to_gray_n_resize(next_state)
            next_state = np.expand_dims(next_state, axis=2)
            next_state = np.expand_dims(next_state, axis=0)
            # TODO: Interchange next and previous states to check changes
            next_states = np.append(next_state, state[:, :, :, :3], axis=3)
            replay_memory.append((state, action, reward, next_states, done))
            state = next_states
            if done:
                observation = env.reset()
                observation = ops.convert_to_gray_n_resize(observation)
                observation = np.expand_dims(observation, axis=2)
                state = np.repeat(observation, 4, axis=2)
                state = np.expand_dims(state, axis=0)
            print("Random Observation: {}/{}".format(i+1, mc.rand_observation_time))
        print("Total Positive rewards: {}".format(r))
    return replay_memory


def make_directories(main_dir):
    main_dir = main_dir + "Time_{}_{}_{}".format(datetime.datetime.now(), mc.n_epochs, mc.learning_rate)
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
            action = np.argmax(sess.run(agent, feed_dict={X_input: state}))
            if show_action:
                print(action)
            new_state, r, done, _ = env.step(action)
            # r = ops.convert_reward(r)
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

    # loss = tf.reduce_mean(tf.square(agent - Y_target))
    loss = tf.losses.mean_squared_error(labels=Y_target, predictions=agent)

    # TODO: Add loss decay operation
    optimizer = tf.train.RMSPropOptimizer(learning_rate=mc.learning_rate, epsilon=0.01, momentum=0.95).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=mc.learning_rate).minimize(loss)

    # Create the summary for tensorboard
    tf.summary.scalar(name='loss', tensor=loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        if train_model:
            print("Training agent!")
            tensorboard_dir, saved_model_dir, log_dir = make_directories(mc.logdir)
            print("Tensorboard files stores in: {}".format(tensorboard_dir))
            print("Saved models stored in: {}".format(saved_model_dir))
            print("Log files stores in: {}".format(log_dir))
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_dir, graph=sess.graph)
            t1 = time.time()
            step = 0
            total_steps = mc.n_epochs * mc.observation_time
            prob_rand = mc.prob_random

            # Replay memory
            replay_memory = deque()
            replay_memory = collect_rand_observations(replay_memory)  # Get the initial 50k random observations
            observation = env.reset()
            observation = ops.convert_to_gray_n_resize(observation)
            observation = np.expand_dims(observation, axis=2)
            state = np.repeat(observation, 4, axis=2)
            state = np.expand_dims(state, axis=0)

            # Save current model parameters
            with open("mission_control.py", "r") as mc_file:
                mission_control_file = mc_file.read()
                with open(log_dir + "/mission_control.txt", "w") as mc_writer:
                    mc_writer.write(mission_control_file)

            for e in range(mc.n_epochs):
                print("--------------------------Epoch: {}/{}------------------------------".format(e + 1, mc.n_epochs))
                with open(log_dir + "/log.txt", "a") as log_file:
                    log_file.write("--------------------------Epoch: {}/{}------------------------------\n".format(e + 1, mc.n_epochs))
                for b in range(int(mc.observation_time)):

                    # Train on the some random steps
                    mini_batch = random.sample(replay_memory, mc.batch_size)
                    agent_input = []
                    agent_target = []
                    for s in range(len(mini_batch)):
                        state_ = mini_batch[s][0]
                        action_ = mini_batch[s][1]
                        reward_ = mini_batch[s][2]
                        next_state_ = mini_batch[s][3]
                        done_ = mini_batch[s][4]

                        agent_input.append(state_[0])
                        target = sess.run(agent, feed_dict={X_input: state_})
                        if done_:
                            target[0][action_] = reward_
                            agent_target.append(target[0])
                        else:
                            agent_output = sess.run(agent, feed_dict={X_input: next_state_})
                            target[0][action_] = reward_ + mc.gamma * (np.amax(agent_output))
                            agent_target.append(target[0])

                    # Training the agent for 1 iterations. Finally!!
                    for i in range(mc.fit_epochs):
                        sess.run(optimizer, feed_dict={X_input: agent_input, Y_target: agent_target})
                    l, summary = sess.run([loss, summary_op], feed_dict={X_input: agent_input, Y_target: agent_target})
                    writer.add_summary(summary, global_step=step)
                    with open(log_dir + "/log.txt", "a") as log_file:
                        log_file.write("Step: {}/{}\n".format(b, int(total_steps)))
                        log_file.write("Loss: {:.10f}\n".format(l))
                    print("Step: {}/{}".format(b, int(total_steps)))
                    print("Loss: {:.10f}\n".format(l))

                    # Collect the next observation
                    if np.random.rand() <= prob_rand:
                        action = np.random.randint(low=0, high=env.action_space.n)
                    else:
                        action = np.argmax(sess.run(agent, feed_dict={X_input: state}))
                    next_state, reward, done, _ = env.step(action)
                    # reward = ops.convert_reward(reward)
                    next_state = ops.convert_to_gray_n_resize(next_state)
                    next_state = np.expand_dims(next_state, axis=2)
                    next_state = np.expand_dims(next_state, axis=0)

                    next_states = np.append(next_state, state[:, :, :, :3], axis=3)

                    # Remove old samples from replay memory if it's full
                    if len(replay_memory) > mc.observation_time:
                        replay_memory.popleft()
                    replay_memory.append((state, action, reward, next_states, done))
                    state = next_states
                    step += 1
                    prob_rand = ops.anneal_epsilon(prob_rand, step)
                    if done:
                        observation = env.reset()
                        observation = ops.convert_to_gray_n_resize(observation)
                        observation = np.expand_dims(observation, axis=2)
                        state = np.repeat(observation, 4, axis=2)
                        state = np.expand_dims(state, axis=0)
                    # Save the agent
                    if (b+1) % 50000 == 0:
                        print("------------------------Saving----------------------------")
                        # play(sess=sess, agent=agent, no_plays=mc.n_plays, log_dir=log_dir, show_ui=mc.show_ui, show_action=mc.show_action)
                        saved_path = saver.save(sess, saved_model_dir + '/model', global_step=step)
                        print("Model saved in: {}".format(saved_path))
                saved_path = saver.save(sess, saved_model_dir + '/model', global_step=step)
            print("Time taken of {} epochs on your potato: {:.4f}s".format(mc.n_epochs, time.time() - t1))
            print("Average time for each epoch: {:.4f}s".format((time.time() - t1) / mc.n_epochs))
            print("Tensorboard files saved in: {}".format(tensorboard_dir))
            print("Model saved in: {}".format(saved_path))
            print("Model parameters stored in: {}".format(log_dir + "mission_control.txt"))
            print("Agent get to roll!")
            with open(log_dir + "/log.txt", "a") as log_file:
                log_file.write("Time taken of {} epochs on your potato: {:.4f}s\n".format(mc.n_epochs, time.time() - t1))
                log_file.write("Average time for each epoch: {:.4f}s\n".format((time.time() - t1) / mc.n_epochs))
        else:
            # Get the latest trained model
            saved_models = os.listdir(mc.logdir)
            latest_saved_model = sorted(saved_models)[-1]
            saver.restore(sess, tf.train.latest_checkpoint(mc.logdir + latest_saved_model + "/saved_models/"))
            print("Getting model from: {}".format(mc.logdir + latest_saved_model + "/saved_models/"))
            print("------------------------Playing----------------------------")

            play(sess=sess, agent=agent, no_plays=mc.n_plays, log_dir=None, show_ui=mc.show_ui, show_action=mc.show_action)


if __name__ == '__main__':
    train(train_model=mc.train_model)
