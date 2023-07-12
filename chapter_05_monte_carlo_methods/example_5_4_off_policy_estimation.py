'''
In progress
'''

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from typing import Any
from tqdm import tqdm


# Get probability of the action given a state from target policy
def prob_target_policy(player_sum:int, action:int) -> float:
    if player_sum >= 20 and action == 0:
        return 1.
    if player_sum < 20 and action == 1:
        return 1.
    return 0.

# Get probability of the action given a state from the behaviro policy
def prob_behavior_policy():
    return 0.5 # as defined in the book

# get random action
def get_action() -> int:
    return np.random.choice([0, 1], p=[0.5, 0.5])

# Calculate MSE loss between given value and target value
def mse(val_1, val_2):
    return np.sqrt((val_1 - val_2) ** 2)


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
    gamma = 1.0
    n_rounds = 1

    record = {'G': 0., 'tau': 0, 'rho': 0.}
    value_hist = {'ordinary': np.zeros(shape=(n_rounds, total_episodes)),
                  'weighted': np.zeros(shape=(n_rounds, total_episodes))}

    env = gym.make('Blackjack-v1', sab=True)


    for r in range(n_rounds):

        V_ord = 0
        V_wei = 0
        C = 0
        tau = 0

        for t in tqdm(range(total_episodes)):
            terminated = False
            env, observation = init_start_state(env, start_state)
            state = observation # for uniform naming purpose
            action = get_action()
            
            traj = []
            seen = set()
            
            # Get a trajectory
            while not terminated:
                observation, reward, terminated, truncated, info = env.step(action)
                traj.append((state, action, reward))
                state = observation
                action = get_action()
            
            G = 0
            W = 1.0
        
            while traj:                
                state, action, reward = traj.pop()
                G = gamma * G + reward

                if state == start_state and state not in seen:                    
                    
                    seen.add(state)
                    
                    V_ord += W * G
                    tau += 1
                    C = C + W
                    
                    if C != 0:
                        V_wei = V_wei + (W / C) * (G - V_wei)

                prob_pi = prob_target_policy(state[0], action)
                prob_behav = prob_behavior_policy()
                W = W * (prob_pi / prob_behav)


            mse_wei = mse(V_wei, target_val)
            mse_ord = mse(V_ord / tau, target_val)
            value_hist['weighted'][r, t] = mse_wei
            value_hist['ordinary'][r, t] = mse_ord

    ord_hist = value_hist['ordinary'].mean(axis=0)
    weighted_hist = value_hist['weighted'].mean(axis=0)

    x = np.arange(total_episodes)
    plt.xscale('log')
    plt.plot(x, weighted_hist, label='wei')
    plt.plot(x, ord_hist, label='ord')
    plt.legend()
    plt.show()