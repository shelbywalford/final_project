# final_project

This is a slightly confusing installation process, but here it goes. First, the official reinforcement learning to run github repository
is found here (https://github.com/stanfordnmbl/osim-rl). The README includes the first set of installation instructions: follow the
directions up until the "Basic Usage" section.

Then, I would suggest copying and pasting the following code into the command line to make sure things are running as they should be:
###
from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)
observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
###

This includes the basic model that we are training to run. Next, the github repository that includes the training algorithms we are
testing (Keras and TensorForce) have been uploaded here: https://github.com/seungjaeryanlee/osim-rl-helper
Download this repository, and then you can test to make sure it's working correctly by typing the following code into the command line:
###
./run.py RandomAgent -v
###

Now the model reward can be edited and trained to actually run!
