import numpy as np
import tensorflow as tf
import gym
import time
import os
import datetime
import random
from collections import deque
import mission_control_lunar as mc
import ops
import sys
import itertools

# Setup the environment
env = gym.make('LunarLander-v2')

# Placeholders
# TODO: make the shape of X_input generalized
X_input = tf.placeholder(dtype=tf.float32, shape=[None, env.observation_space.shape[0] * 2], name='Observations')
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, env.action_space.n], name='Target_Q_values')


# Dense Fully Connected Agent
def get_agent(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    dense_1 = tf.nn.relu(ops.dense(x, env.observation_space.shape[0] * 2, mc.dense_1, name='dense_1'))
    dense_2 = tf.nn.relu(ops.dense(dense_1, mc.dense_1, mc.dense_2, name='dense_2'))
    dense_3 = tf.nn.relu(ops.dense(dense_2, mc.dense_2, mc.dense_3, name='dense_3'))
    output = ops.dense(dense_3, mc.dense_3, env.action_space.n, name='output')
    return output


def collect_rand_observations(replay_memory):
    print("Collecting Random Observations")
    observation = env.reset()
    observation = np.expand_dims(observation, axis=0)
    state = np.append(observation, observation, 1)
    for i in range(int(mc.rand_observation_time)):
        action = np.random.randint(low=0, high=env.action_space.n)
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_states = np.expand_dims(np.append(next_state, state[0][:env.observation_space.shape[0]]), axis=0)
        replay_memory.append((state, action, reward, next_states, done))
        state = next_states
        if done:
            observation = env.reset()
            observation = np.expand_dims(observation, axis=0)
            state = np.append(observation, observation, 1)
        print("\rRandom Observation: {}/{}".format(i + 1, mc.rand_observation_time), end="")
    return replay_memory


def make_directories(main_dir):
    main_dir = main_dir + "Time_{}_{}_{}".format(datetime.datetime.now(), mc.n_plays, mc.learning_rate)
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    tensorboard_dir = main_dir + "/Tensorboard"
    saved_model_dir = main_dir + "/saved_models"
    log_dir = main_dir + "/logs"
    os.mkdir(tensorboard_dir)
    os.mkdir(saved_model_dir)
    os.mkdir(log_dir)
    return tensorboard_dir, saved_model_dir, log_dir


def play(sess, agent, no_plays, show_ui=False, show_action=False):
    rewards = []
    for p in range(no_plays):
        observation = env.reset()
        observation = np.expand_dims(observation, axis=0)
        state = np.append(observation, observation, 1)
        done = False
        reward = []
        while not done:
            if show_ui:
                env.render()
            action = np.argmax(sess.run(agent, feed_dict={X_input: state}))
            if show_action:
                print(action)
            new_state, r, done, _ = env.step(action)
            new_state = np.expand_dims(new_state, axis=0)
            state = np.expand_dims(np.append(new_state, state[0][:env.observation_space.shape[0]]), axis=0)
            reward.append(r)
        rewards.append(np.sum(reward))
        print("\rGame: {}/{}".format(p + 1, no_plays))
        print("\rReward: {}".format(np.sum(reward)))
        print("Last reward: {}".format(reward[-3:]))
        sys.stdout.flush()
    print("------------------------------------------------------------------------------------------------------")
    print("Best reward: {}".format(np.amax(rewards)))
    print("Average reward: {}".format(np.mean(rewards)))


def train(train_model=True):
    agent = get_agent(X_input)

    # loss = tf.reduce_mean(tf.square(agent - Y_target))
    loss = tf.losses.mean_squared_error(labels=Y_target, predictions=agent)

    # TODO: Add loss decay operation
    optimizer = tf.train.AdamOptimizer(learning_rate=mc.learning_rate).minimize(loss)

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
            tensorboard_dir, saved_model_dir, log_dir = make_directories(mc.logdir)
            print("Tensorboard files stores in: {}".format(tensorboard_dir))
            print("Saved models stored in: {}".format(saved_model_dir))
            print("Log files stores in: {}".format(log_dir))
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_dir, graph=sess.graph)
            t1 = time.time()
            step = 0
            prob_rand = mc.prob_random

            # Replay memory
            replay_memory = deque()
            replay_memory = collect_rand_observations(replay_memory)  # Get the initial 50k random observations

            for e in range(mc.n_plays):
                print("--------------------------Play: {}/{}------------------------------\n".format(e + 1, mc.n_plays))
                with open(log_dir + "/log.txt", "a") as log_file:
                    log_file.write("--------------------------Play: {}/{}------------------------------\n".format(e + 1, mc.n_plays))

                observation = env.reset()
                observation = np.expand_dims(observation, axis=0)
                state = np.append(observation, observation, 1)

                episode_rewards = []
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

                        agent_input.append(state_[0])
                        target = sess.run(agent, feed_dict={X_input: state_})
                        if done_:
                            target[0][action_] = reward_
                            agent_target.append(target[0])
                        else:
                            agent_output = sess.run(agent, feed_dict={X_input: next_state_})
                            target[0][action_] = reward_ + mc.gamma * (np.amax(agent_output))
                            agent_target.append(target[0])

                    # Training the agent for 10 iterations. Finally!!
                    for i in range(mc.fit_epochs):
                        sess.run(optimizer, feed_dict={X_input: agent_input, Y_target: agent_target})
                    l, summary = sess.run([loss, summary_op], feed_dict={X_input: agent_input, Y_target: agent_target})
                    writer.add_summary(summary, global_step=step)

                    print("\rStep: {} ({}), Play: {}/{}, Loss: {}".format(t, step, e+1, mc.n_plays, l), end="")
                    sys.stdout.flush()

                    if np.random.rand() < prob_rand:
                        action = np.random.randint(low=0, high=env.action_space.n)
                    else:
                        q_prediction = sess.run(agent, feed_dict={X_input: state})
                        action = np.argmax(q_prediction)
                        log_q_values.extend(q_prediction)

                    next_state, reward, done, _ = env.step(action)
                    next_state = np.expand_dims(next_state, axis=0)
                    # TODO: Interchange next and previous states to check changes
                    next_states = np.expand_dims(np.append(next_state, state[0][:env.observation_space.shape[0]]), axis=0)

                    # Remove old samples from replay memory if it's full
                    if len(replay_memory) > mc.observation_time:
                        replay_memory.popleft()

                    replay_memory.append((state, action, reward, next_states, done))
                    episode_rewards.append(reward)
                    if mc.show_ui:
                        env.render()
                    state = next_states
                    step += 1

                    if done:
                        break

                with open(log_dir + "/log.txt", "a") as log_file:
                    log_file.write("Step: {} ({}), Play: {}/{}, Loss: {}\n".format(t, step, e + 1, mc.n_plays, l))
                    log_file.write("Reward Obtained: {}\n".format(np.sum(episode_rewards)))
                    log_file.write("Last reward: {}\n".format(episode_rewards[-4:]))
                    if log_q_values != []:
                        log_file.write("Average Q Value: {}\n".format(np.mean(log_q_values)))
                    else:
                        log_file.write("All of the actions were random\n")

                if log_q_values != []:
                    print("\nAverage Q Value: {}".format(np.mean(log_q_values)))
                else:
                    print("\nAll of the actions were random")

                prob_rand = ops.anneal_epsilon(prob_rand, step)
                print("Reward Obtained: {}".format(np.sum(episode_rewards)))
                print("Last reward: {}".format(episode_rewards[-4:]))
                print("Random Move Prob: {}".format(prob_rand))

                # Save the agent
                if (e+1) % 100 == 0:
                    print("------------------------Playing----------------------------")
                    play(sess, agent, mc.n_actual_plays, mc.show_ui)
                    saved_path = saver.save(sess, saved_model_dir + '/model_{}'.format(datetime.datetime.now()))
            print("Time taken of {} epochs on your potato: {:.4f}s".format(mc.n_plays, time.time() - t1))
            print("Average time for each epoch: {:.4f}s".format((time.time() - t1) / mc.n_plays))
            print("Tensorboard files saved in: {}".format(tensorboard_dir))
            print("Model saved in: {}".format(saved_path))
            print("Model parameters stored in: {}".format(log_dir + "mission_control.txt"))
            print("Agent get to roll!")
            with open(log_dir + "/log.txt", "a") as log_file:
                log_file.write("Time taken of {} epochs on your potato: {:.4f}s\n".format(mc.n_plays, time.time() - t1))
                log_file.write("Average time for each epoch: {:.4f}s\n".format((time.time() - t1) / mc.n_plays))
            with open("mission_control_lunar.py", "r") as mc_file:
                mission_control_file = mc_file.read()
                with open(log_dir + "/mission_control.txt", "w") as mc_writer:
                    mc_writer.write(mission_control_file)
        else:
            # Get the latest trained model
            saved_models = os.listdir(mc.logdir)
            latest_saved_model = sorted(saved_models)[-1]
            saver.restore(sess, tf.train.latest_checkpoint(mc.logdir + latest_saved_model + "/saved_models/"))
            print("Getting model from: {}".format(mc.logdir + latest_saved_model + "/saved_models/"))
            print("------------------------Playing----------------------------")
            play(sess, agent, mc.n_plays, mc.show_ui, mc.show_action)


if __name__ == '__main__':
    train(train_model=mc.train_model)
