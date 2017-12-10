import numpy as np
import tensorflow as tf
import gym
import os
import mission_control_breakout as mc
import ops
import matplotlib.pyplot as plt
from gym.wrappers import Monitor

# Setup the environment
env = gym.make('BreakoutDeterministic-v4')
# env = Monitor(env=env, directory="./Results/Videos/Breakout", resume=True)

# Placeholders
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name='Observations')
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='Target_Q_values')


def get_agent(x, reuse=False):
    """
    Generate the CNN agent
    :param x: tensor, Input frames concatenated along axis 3
    :param reuse: bool, True -> Reuse weight variables
                        False -> Create new ones
    :return: Tensor, logits for each valid action
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    x = tf.divide(x, 255.0, name='Normalize')
    conv_1 = tf.nn.relu(ops.cnn_2d(x, weight_shape=mc.conv_1, strides=mc.stride_1, name='conv_1'))
    conv_2 = tf.nn.relu(ops.cnn_2d(conv_1, weight_shape=mc.conv_2, strides=mc.stride_2, name='conv_2'))
    conv_3 = tf.nn.relu(ops.cnn_2d(conv_2, weight_shape=mc.conv_3, strides=mc.stride_3, name='conv_3'))
    conv_3_r = tf.reshape(conv_3, [-1, 7 * 7 * 64], name='reshape')
    dense_1 = tf.nn.relu(ops.dense(conv_3_r, 7 * 7 * 64, mc.dense_1, name='dense_1'))
    output = ops.dense(dense_1, mc.dense_1, mc.dense_2, name='dense_2')
    return output


def make_directories():
    """
    Create directories to store tenorboard files, saved models and log files during each unique run.
    :param main_dir: String, points to a results file
    :return: list of strings, required directories paths
    """
    main_dir = "./Dataset/Breakout"
    train_dir = main_dir + "/train"
    test_dir = main_dir + "/test"
    os.mkdir(main_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    return main_dir, train_dir, test_dir


def play_n_collect(sess, agent, no_plays, log_dir=None, show_ui=False, show_action=False):
    """
    Use a trained agent to play a required number of games
    :param sess: op, session instance from tensorflow
    :param agent: tensor, trained agent structure/graph
    :param no_plays: int, you get it
    :param log_dir: string, place to store the log files during gameplay
    :param show_ui: bool, True  -> Show game screen
                          False -> Should I explain this?
    :param show_action: bool, True  -> Show the actions taken by the trained agent
                              False -> Hmm, what can this be?
    :return: just prints the results with nothing being returned
    """
    rewards = []
    main_dir, train_dir, test_dir = make_directories()
    step = 0
    for p in range(no_plays):
        frame = 0
        observation = env.reset()
        if p < 1000:
            # Save the first image
            episode_path = train_dir + "/{:05d}".format(p)
        else:
            episode_path = test_dir + "/{:05d}".format(p % 1000)
        os.mkdir(episode_path)
        plt.imsave(arr=observation, fname=episode_path + "/{:06d}.png".format(frame))

        observation = ops.convert_to_gray_n_resize(observation)
        observation = np.expand_dims(observation, axis=2)
        state = np.repeat(observation, 4, axis=2)
        state = np.expand_dims(state, axis=0)
        done = False
        reward = 0
        while not done:
            if show_ui:
                env.render()
            if np.random.rand() < 0.07:
                action = env.action_space.sample()
            else:
                action = np.argmax(sess.run(agent, feed_dict={X_input: state}))

            # Save the action taken
            with open(episode_path + "/action.txt", "a") as log:
                log.write("{}\n".format(action))

            if show_action:
                print(action)
            frame += 1
            step += 1
            new_state, r, done, _ = env.step(action)
            plt.imsave(arr=new_state, fname=episode_path + "/{:06d}.png".format(frame))
            next_state = ops.convert_to_gray_n_resize(new_state)
            next_state = np.expand_dims(next_state, axis=2)
            next_state = np.expand_dims(next_state, axis=0)
            state = np.append(next_state, state[:, :, :, :3], axis=3)
            reward += r
        rewards.append(reward)
        print("Step: {}/500e3".format(step))
        print("Game: {}/{}".format(p + 1, no_plays))
        print("Reward: {}\n".format(reward))
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


def train():
    """
    Trains the agent with hyperparameters and other info loaded from mission_control_<game>.py file
    :param train_model: bool, True  -> Trains the agent
                              False -> Loads the LATEST trained agent and plays
    :return: absolutely nothing
    """
    with tf.variable_scope("Action_agent"):
        agent = get_agent(X_input)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Get the latest trained model
        saved_models = os.listdir(mc.logdir)
        latest_saved_model = sorted(saved_models)[-1]
        saver.restore(sess, tf.train.latest_checkpoint(mc.logdir + latest_saved_model + "/saved_models/"))
        print("Getting model from: {}".format(mc.logdir + latest_saved_model + "/saved_models/"))
        print("------------------------Playing----------------------------")
        play_n_collect(sess=sess, agent=agent, no_plays=1200, log_dir=None,
                       show_ui=False, show_action=mc.show_action)


if __name__ == '__main__':
    train()
