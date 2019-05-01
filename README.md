# Final Project

# Setup
This is a slightly confusing installation process, but here it goes. First, the official reinforcement learning to run github repository
is found here (https://github.com/stanfordnmbl/osim-rl). The README includes the first set of installation instructions: follow the
directions up until the "Basic Usage" section. Make sure that you have the "opensim-rl" environment activated when you are running any 
future commands.

Then, I would suggest copying and pasting the following code into the command line to make sure things are running as they should be:

    from osim.env import ProstheticsEnv 
    env = ProstheticsEnv(visualize=True)
    observation = env.reset()
    for i in range(200):
        observation, reward, done, info = env.step(env.action_space.sample())


This includes the basic model that we are training to run. Next, the github repository that contains helper files that set up two 
reinforcement learning optimizers (DDPG and PPO) have been uploaded here: https://github.com/seungjaeryanlee/osim-rl-helper. 
To use these methods, you must download the DDPG package "keras" and the PPO package "TensorForce". To install these packages, 
use the commands:

    conda install keras
    pip install keras-rl
    pip install TensorForce

# Training and Visualization

After downloading the repository, you can test to make sure it's working correctly by typing the following code into the command line:

    ./run.py RandomAgent -v

This agent produced random muscle forces and does not use reinforcement learning. You can also train the Keras and TensorForce 
agents using the commands:

    ./run.py KerasDDPGAgent --train 10000
or,
    ./run.py TensorForcePP0Agent --train 10000

Where "--train #" specifies the number of training steps.

Now the optimization parameters and reward function can can be edited to train the model to walk! To visualize the simulation after 
training an agent, type "-v" after the agent name on the command line.
# Uploaded Files
We're currently making changes to the optimization parameters in the KerasAgentDDPG file, and the reward function in the osim file. 
