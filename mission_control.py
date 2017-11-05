########################################################################################################################
# Training
learning_rate = 0.00025
batch_size = 100
observation_time = 5e5
rand_observation_time = 5e4
prob_random = 1
gamma = 0.9
n_epochs = 5
fit_epochs = 1
n_plays = 5  # TODO: Change to 100

########################################################################################################################
# Agent Model
# dense_1 = 40
# dense_2 = 18
# dense_3 = 10
conv_1 = [8, 8, 4, 32]
stride_1 = [1, 4, 4, 1]
conv_2 = [4, 4, 32, 64]
stride_2 = [1, 2, 2, 1]
conv_3 = [3, 3, 64, 64]
stride_3 = [1, 1, 1, 1]
dense_1 = 256
dense_2 = 4


########################################################################################################################
# Control
train_model = False
show_ui = True
show_action = True

########################################################################################################################
# Paths
logdir = "./Results/Breakout/"  # Use: "./Results/CartPole/", "./Results/Breakout/"
