import numpy as np
import gymnasium as gym
from collections import defaultdict



def initialize_policy(shape:tuple) -> np.ndarray:
    policy = np.zeros(shape=shape) # (player_sums, dealer_faces)
    # Following the description of ex. 5.3, the initial policy
    # is to draw when player sums smaller than 20
    policy[:20, :] = 1
    return policy




if __name__ == "__main__":

    env = gym.make('Blackjack-v1', sab=True)
    player_sums = env.observation_space[0].n
    dealer_faces = env.observation_space[1].n
    actions = env.action_space.n

    # Initialize action-value function 
    Q = np.zeros(shape=(player_sums, dealer_faces, actions))
    policy = initialize_policy(shape=(player_sums, dealer_faces))
    Returns = defaultdict(list)
    