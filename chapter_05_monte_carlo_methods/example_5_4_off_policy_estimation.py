'''
In progress
'''

import numpy as np
import gymnasium as gym
from collections import defaultdict
from matplotlib import pyplot as plt
from typing import Any


# Get probability of the action given a state from target policy
def prob_target_policy(player_sum:int, action:int) -> float:
    raise NotImplementedError

# Get probability of the action given a state from the behaviro policy
def prob_behavior_policy():
    raise NotImplementedError

# Calculate state-transition probability
def calc_rho(state_action_pair: list):
    raise NotImplementedError

# initialize the state to the preset start state
def init_start_state(env:Any, start_state:tuple) -> tuple:
    observation, info = env.reset()
    while observation != start_state:
        observation, info = env.reset()
    return env, observation


if __name__ ==  "__main__":
    start_state = (13, 2, 1) # player sum = 13, dealer face = 2, usable ace = 1
    target_val = -0.27726 # Provided in the book
    total_episodes = 1_000

    # Initialize 
    Tau = set()
    rho = defaultdict(float)
    G = defaultdict(float)

    env = gym.make('Blackjack-v1', sab=True)
    env, observation = init_start_state(env, start_state)
