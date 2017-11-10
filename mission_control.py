########################################################################################################################
# Training
learning_rate = 0.001
batch_size = 50
observation_time = 5e3
rand_observation_time = 5e2
prob_random = 0.9
gamma = 0.9
n_epochs = 1000
fit_epochs = 10
n_plays = 500  # TODO: Change to 100
n_actual_plays = 100

########################################################################################################################
# Agent Model
dense_1 = 40
dense_2 = 18
dense_3 = 10
"""
conv_1 = [8, 8, 4, 32]
stride_1 = [1, 4, 4, 1]
conv_2 = [4, 4, 32, 64]
stride_2 = [1, 2, 2, 1]
conv_3 = [3, 3, 64, 64]
stride_3 = [1, 1, 1, 1]
dense_1 = 512  # Was 256 before
dense_2 = 4
"""

########################################################################################################################
# Control
train_model = True
show_ui = False
show_action = False

########################################################################################################################
# Paths
logdir = "./Results/CartPole_v1/"  # Use: "./Results/CartPole/", "./Results/Breakout/"
