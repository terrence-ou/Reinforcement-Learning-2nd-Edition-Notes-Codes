import numpy as np
import gymnasium as gym

if __name__ == "__main__":
    # Create blackjack environment
    # Turn sab True to make the env follows the S&B books reward
    env = gym.make("Blackjack-v1", sab=True)
