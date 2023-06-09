import numpy as np
import gymnasium as gym
from collections import defaultdict
import utils

# In example 5.1, the policy is:
# - stick (0) if the player's sum is 20 or 21
# - hits (1) otherwise
def policy(player_curr_sum:int) -> int:
    return 0 if player_curr_sum >= 20 else 1


if __name__ == "__main__":
    # Create blackjack environment
    # Turn sab True to make the env follows the S&B books reward
    env = gym.make("Blackjack-v1", sab=True)
    # Observation -> (player_sum, dealer_showing, usable_ace)
    player_sums = env.observation_space[0].n
    dealer_faces = env.observation_space[1].n
    aces = env.observation_space[2].n

    # Initialize V
    V = np.zeros(shape=(player_sums, dealer_faces, aces))

    # # Initialize Returns
    # # a. List method stated in the book:
    # # Returns = defaultdict(list)
    # # b. Incremental update method, way faster than the list method:
    Returns = np.zeros_like(V, dtype=np.float32)
    counts = np.zeros_like(V, dtype=np.int16)

    # # Hyper parameters
    num_episodes = 500_000
    gamma = 1.0

    # # Loop over episodes
    for _ in range(num_episodes):
        terminated = False
        observation, info = env.reset()
        state = observation
        action = policy(state[0])

        traj = [] # trajectory of (state, action, reward)
        seen = set() # For first-visit MC method

        # Get a trajectory
        while not terminated:
            observation, reward, terminated, truncated, info = env.step(action)
            traj.append((state, action, reward))
            # Get next state and next action
            state = observation
            action = policy(state[0])
        
        G = 0.
        while traj:
            state, _, reward = traj.pop()
            G = gamma * G + reward
            if state not in seen:
                seen.add(state)
                # a. For the list method:                
                # Returns[state].append(G)
                # V[state] = np.mean(Returns[state])
                # b. For the incremental update method:
                Returns[state] += G
                counts[state] += 1
                V[state] = Returns[state] / counts[state]

    # Visualize results
    for usable_ace in range(2):
        if_useable_ace = 'usable ace' if usable_ace else 'no usable ace'
        postfix = '_'.join(if_useable_ace.split(' '))
        file_path_name = './plots/example_5_1/' + f'{num_episodes}_episodes_' + postfix + '.png'
        utils.plot_surface(V[:, :, usable_ace], 
                           title=f'After {num_episodes} episodes ({if_useable_ace})', 
                           savefig=True, 
                           file_path_name=file_path_name)
