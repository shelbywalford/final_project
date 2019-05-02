from tensorforce.agents import PPOAgent
from helper.wrappers import ClientToEnv, DictToListFull, JSONable

from ...templates import TensorforceAgent


class TensorforcePPOAgent(TensorforceAgent):
    def __init__(self, observation_space, action_space,
                 directory='./TensorforcePPOAgent/'):
        
        # Create a Proximal Policy Optimization agent:
        # This agent is restricted to a 0 or 1 activation. To enable continuous activations, change the action type to "float" and delete "num_actions".
        
        self.agent = PPOAgent(
            states=dict(type='float', shape=observation_space.shape),
            actions=dict(type='int', shape=action_space.shape,num_actions=2,
                         min_value=0, max_value=1),
            # This PPO Agent neural network has two dense hidden layers with 256 nodes.
            network=[
                dict(type='dense', size=256),
                dict(type='dense', size=256),
            ],
            
            # The agent uses an "Adam" optimizer with a learning rate of .0001
            batching_capacity=1000,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-4
            )
        )
        self.directory = directory
