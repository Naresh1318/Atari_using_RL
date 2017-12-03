########################################################################################################################
# Training
learning_rate = 0.00025
batch_size = 32
observation_time = 1e6  # 1e3
rand_observation_time = 5e4  # 5e2
target_network_update = 1e4  # 1e3
prob_random = 1
gamma = 0.99
n_episodes = 1e4  # 5
fit_epochs = 1
weight_init = 0.01
momentum = 0.95
epsilon = 0.01


########################################################################################################################
# Agent Model
conv_1 = [8, 8, 4, 32]
stride_1 = [1, 4, 4, 1]
conv_2 = [4, 4, 32, 64]
stride_2 = [1, 2, 2, 1]
conv_3 = [3, 3, 64, 64]
stride_3 = [1, 1, 1, 1]
dense_1 = 512
dense_2 = 4


########################################################################################################################
# Control
train_model = True
show_ui = False
show_action = False

########################################################################################################################
# Paths
logdir = "./Results/Breakout/"  # Use: "./Results/CartPole/", "./Results/Breakout/"
