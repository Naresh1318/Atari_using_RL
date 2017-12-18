# Playing Atari the right way!

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/Breakout_ram_good.gif" alt="Cover" width="300px"/>
</p>

<br>
The simplest implementation of playing Atari games using game screen as input. Also contains code to 
implement visual foresight using adversarial action conditioned video prediction model (Still working on this).

**Paper**: 
[Playing Atari the right way!, ECE6504 Project, 2017](https://drive.google.com/open?id=1s2jKSDQGXy4xC0-SCnFvydPdf8DaE-8g)


## Dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 atari

 Install dependencies

    pip3 install -r requirements.txt
    
***Notes:***
* Training the agent to play breakout at a reasonable level took about 80 hours on two p100s. 
Don't even think about running this on a CPU. **I would highly appreciate it if you can submit a pull request that 
makes training faster** (I know some of my methods suck).

* The trained models can easily be used to test the performance of an agent on a CPU.

## Architecture graph from Tensorboard

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/Architectur.jpg" alt="Cartpole agent">
</p>

## Training a DQN agent
### Playing Cartpole using the game states as input

    python3 play_cartpole.py

To change the hyperparameters modify `mission_control_cartpole.py`.

<p align="center">
<img src="ontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/CartPole.gif" alt="Cartpole agent">
</p>

**Note:**
* This isn't as computationally demanding as Breakout using frames.

### Playing Breakout using the game frames as input

    python3 play_breakout.py

To change the hyperparameters modify `mission_control_breakout.py`.

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/Breakout_ram_good.gif" width="300px" alt="Breakout agent">
</p>

**Note:**
* I have included the trained model for Breakout after 14 million episodes. Just explore the Results director for Breakout.
* Change `train_model` to `False` and `show_ui` to `True` to load the saved model and see the agent in action.


## Results from training Breakout agent

### Plot of the rewards obtained per episode during training

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/Breakout_rewards.png" alt="Breakout Reward" width="500px">
</p>

### Q-value histogram after each episode

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/q_val_hist.jpg" alt="Breakout histo">
</p>


### Max Q-values after each episode

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/max_q_value.jpg" alt="Breakout max Q">
</p>

## Use the trained model to generate dataset

    python3 generate_dataset.py
    
**Note:**

* You might get some directory nt found errors (Will fix it soon) or just figure it out.

## Training an action conditioned video prediction model

    python3 generate_model_skip.py

**Note:**

* This uses the adversarial action conditioned video prediction model.
* Run `generate_model.py` to use the architecture from [2].


## Results from action conditioned video prediction model

<p align="center">
<img src="https://raw.githubusercontent.com/Naresh1318/Playing_Atari_the_right_way/master/README/skip.jpg" alt="Breakout agent">
</p>
<br>

***Note:***
* Each run generates a required tensorboard files under `./Results/<model>/<time_stamp_and_parameters>/Tensorboard` directory.
* Use `tensorboard --logdir <tensorboard_dir>` to look at loss variations, rewards and a whole lot more.
* Windows gives an error when `:` is used during folder naming (this is produced during the folder creation for each run). I 
would suggest you to remove the time stamp from `folder_name` variable in the `form_results()` function. Or, just dual boot linux!


## References
[1] [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)

[2] [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)
