########################################################################################################################
# Training
learning_rate = 0.00025
batch_size = 32
observation_time = int(1e6)  # 1e3
rand_observation_time = int(5e4)  # 5e2
target_network_update = int(1e4)  # 1e3
prob_random = 1
gamma = 0.99
n_episodes = int(2e4)  # 5
fit_epochs = 1
weight_init = 0.01
momentum = 0.95
epsilon = 0.01

# TODO: Added Stochastic env
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
