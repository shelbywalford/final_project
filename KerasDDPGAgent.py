from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam, RMSprop
import numpy as np
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from ...templates import KerasAgent


class KerasDDPGAgent(KerasAgent):
    """
    An DDPG agent using Keras library with Keras RL.

    For more details about Deep Deterministic Policy Gradient algorithm, check
    "Continuous control with deep reinforcement learning" by Lillicrap.
    https://arxiv.org/abs/1509.02971
    """
    def __init__(self, observation_space, action_space, filename='KerasDDPGAgent.h5f'):
        #from keras.layers.normalization import BatchNormalization       
        nb_actions = action_space.shape[0]

        # Actor network
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + observation_space.shape))
        actor.add(Dense(32),activation='selu'))
        actor.add(Dense(32),activation='selu')
        actor.add(Dense(32),activation='selu')
        actor.add(Dense(nb_actions),activation='sigmoid')      
        print(actor.summary())

        # Critic network
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = concatenate([action_input, flattened_observation])
        x = Dense(64)(x)
        x = Activation('tanh')(x)
        x = Dense(64)(x)
        x = Activation('tanh')(x)
        x = Dense(64)(x)
        x = Activation('tanh')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())

        # Setup Keras RL's DDPGAgent
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.1, mu=0., sigma=.2, 
                dt=1e-2,size=nb_actions,sigma_min=.05,n_steps_annealing=1e6)
        self.agent = DDPGAgent(nb_actions=nb_actions,
                          actor=actor,
                          critic=critic,
                          critic_action_input=action_input,
                          memory=memory,
                          nb_steps_warmup_critic=100,
                          nb_steps_warmup_actor=100,
                          random_process=random_process,
                          gamma=.96,
                          target_model_update=1e-4,
                          delta_clip=1.)
        #opt=RMSprop(lr=0.001,decay=0)
        opt=Adam(lr=3e-4,clipnorm=1.)
        self.agent.compile(optimizer=opt,metrics=['mae'])

        self.filename = filename
