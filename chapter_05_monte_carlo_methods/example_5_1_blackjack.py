
##############################################
# This example is still in progress
##############################################

import numpy as np
import gymnasium as gym

# In example 5.1, the policy is:
# - stick (0) if the player's sum is 20 or 21
# - hits (1) otherwise
def policy(curr_sum:int) -> int:
    return 0 if curr_sum >= 20 else 1


if __name__ == "__main__":
    # Create blackjack environment
    # Turn sab True to make the env follows the S&B books reward
    env = gym.make("Blackjack-v1", sab=True)
    # Observation -> (player_sum, dealer_showing, usable_ace)
    player_sums = env.observation_space[0].n
    dealer_faces = env.observation_space[1].n
    
    # Initialize V
    V = np.zeros(shape=(player_sums, dealer_faces))
    # Initialize Returns
    returns = []
    num_episodes = 10_000


    
    observation, info = env.reset()
