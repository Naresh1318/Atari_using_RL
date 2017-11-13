########################################################################################################################
# Training
learning_rate = 0.0001
batch_size = 32
observation_time = 5e5
rand_observation_time = 5e4
prob_random = 1
gamma = 0.9
fit_epochs = 10
n_plays = 10000  # TODO: Change to 100
n_actual_plays = 100

########################################################################################################################
# Agent Model
dense_1 = 512
dense_2 = 256
dense_3 = 64

########################################################################################################################
# Control
train_model = True
show_ui = False
show_action = False

########################################################################################################################
# Paths
logdir = "./Results/LunarLander/"  # Use: "./Results/CartPole/", "./Results/Breakout/" CartPole_v1
