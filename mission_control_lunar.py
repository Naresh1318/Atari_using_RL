########################################################################################################################
# Training
learning_rate = 0.001
batch_size = 32
observation_time = 5e4
rand_observation_time = 5e3
prob_random = 1
gamma = 0.98
fit_epochs = 1
n_plays = 1000  # TODO: Change to 100
n_actual_plays = 10

# TODO: Have Done.   Changed the init to Glorot init instead of truncated norm

########################################################################################################################
# Agent Model
dense_1 = 40
dense_2 = 20
dense_3 = 10

########################################################################################################################
# Control
train_model = True
show_ui = True
show_action = False

########################################################################################################################
# Paths
logdir = "./Results/LunarLander/"  # Use: "./Results/CartPole/", "./Results/Breakout/" CartPole_v1
