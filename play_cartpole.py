import numpy as np
import tensorflow as tf
import gym
import time
import datetime
import random
from collections import deque
import mission_control as mc
import ops

# Setup the environment
env = gym.make('CartPole-v0')

# Placeholders
# TODO: make the shape of X_input generalized
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='Observations')
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, env.action_space.n], name='Target_Q_values')


# Dense Fully Connected Agent
def get_agent(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    dense_1 = ops.dense(x, 8, mc.dense_1, name='dense_1')
    dense_2 = ops.dense(dense_1, mc.dense_1, mc.dense_2, name='dense_2')
    dense_3 = ops.dense(dense_2, mc.dense_2, mc.dense_3, name='dense_3')
    output = ops.dense(dense_3, mc.dense_3, env.action_space.n, name='output')
    return output


def collect_observations(sess, agent):
    replay_memory = deque()
    observation = env.reset()
    observation = np.expand_dims(observation, axis=0)
    state = np.append(observation, observation, 1)
    for i in range(mc.observation_time):
        if np.random.rand() < mc.prob_random:
            action = np.random.randint(low=0, high=env.action_space.n)
        else:
            action = np.argmax(sess.run(agent, feed_dict={X_input: state}))
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        # TODO: Interchange next and previous states to check changes
        next_states = np.expand_dims(np.append(next_state, state[0][4:]), axis=0)
        replay_memory.append((state, action, reward, next_states, done))
        state = next_states
        if done:
            observation = env.reset()
            observation = np.expand_dims(observation, axis=0)
            state = np.append(observation, observation, 1)
    return replay_memory


def make_directories(main_dir):
    pass


def play(sess, agent, no_plays):
    observation = env.reset()
    observation = np.expand_dims(observation, axis=0)
    state = np.append(observation, observation, 1)
    rewards = []
    for p in range(no_plays):
        done = False
        reward = 0
        while not done:
            action = np.argmax(sess.run(agent, feed_dict={X_input: state}))
            new_state, r, done, _ = env.step(action)
            new_state = np.expand_dims(new_state, axis=0)
            state = np.expand_dims(np.append(new_state, state[0][4:]), axis=0)
            reward += r
        rewards.append(reward)
        print("Game: {}/{}".format(p+1, no_plays))
        print("Reward: {}".format(reward))
    print("------------------------------------------------------------------------------------------------------")
    print("Average reward: {}".format(np.mean(rewards)))


def train(train_model=True):
    agent = get_agent(X_input)

    loss = tf.reduce_mean(tf.square(agent - Y_target))

    optimizer = tf.train.AdamOptimizer(learning_rate=mc.learning_rate).minimize(loss)

    # Create the summary for tensorboard
    tf.summary.scalar(name='loss', tensor=loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        if train_model:
            print("Training agent!")
            print("Tensorboard files stores in: {}".format(mc.logdir))
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=mc.logdir, graph=sess.graph)
            t1 = time.time()
            step = 0
            for e in range(mc.n_epochs):
                print("--------------------------Epoch: {}/{}------------------------------".format(e + 1, mc.n_epochs))
                observations = collect_observations(sess, agent)
                for b in range(int(len(observations)/mc.batch_size)):
                    mini_batch = random.sample(observations, mc.batch_size)
                    agent_input = []
                    agent_target = []
                    for s in range(len(mini_batch)):
                        state = mini_batch[s][0]
                        action = mini_batch[s][1]
                        reward = mini_batch[s][2]
                        next_state = mini_batch[s][3]
                        done = mini_batch[s][4]

                        agent_input.append(state[0])
                        target = sess.run(agent, feed_dict={X_input: state})
                        if done:
                            target[0][action] = reward
                        else:
                            agent_output = sess.run(agent, feed_dict={X_input: next_state})
                            target[0][action] = reward + mc.gamma*(np.amax(agent_output))
                            agent_target.append(target[0])

                        # Training the agent for 10 iterations. Finally!!
                        for i in range(10):
                            sess.run(optimizer, feed_dict={X_input: agent_input, Y_target: agent_target})
                        l, summary = sess.run([loss, summary_op],
                                              feed_dict={X_input: agent_input, Y_target: agent_target})
                        writer.add_summary(summary, global_step=step)
                        print("Batch: {}/{}".format(b+1, int(len(observations)/mc.batch_size)))
                        print("Loss: {:.4f}".format(l))
                        step += 1
                # Save the agent
                saved_path = saver.save(sess, mc.logdir + '/model_{}'.format(datetime.datetime.now()))
                print("Time taken of {} epochs on your potato: {:.4f}s".format(mc.n_epochs, time.time() - t1))
                print("Average time for each epoch: {:.4f}s".format((time.time() - t1)/mc.n_epochs))
                print("Tensorboard files saved in: {}".format(mc.logdir))
                print("Model saved in: {}".format(saved_path))
                print("Agent get to roll!")
                # TODO: Get user input for either play or exit session after training and saving the model
        else:
            # TODO: Make the agent play the game
            saver.restore(sess, mc.logdir)
            play(sess, agent, mc.n_plays)

if __name__ == '__main__':
    train(train_model=True)
