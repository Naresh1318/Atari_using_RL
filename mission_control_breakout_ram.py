########################################################################################################################
# Training
learning_rate = 0.00025
batch_size = 32
observation_time = int(1e6)  # 1e3
rand_observation_time = int(5e4)  # 5e2
target_network_update = int(1e4)  # 1e3
prob_random = 1
gamma = 0.99
n_episodes = int(2e4)  # 10
fit_epochs = 1
weight_init = 0.01
momentum = 0.95
epsilon = 0.01


########################################################################################################################
# Agent Model
dense_1 = 400
dense_2 = 180
dense_3 = 100


########################################################################################################################
# Control
train_model = True
show_ui = True
show_action = False


########################################################################################################################
# Paths
logdir = "./Results/Breakout_ram/"  # Use: "./Results/CartPole/", "./Results/Breakout/"
