
##############################################
# This example is still in progress
##############################################

import numpy as np
import gymnasium as gym
from collections import defaultdict
from matplotlib import pyplot as plt

# In example 5.1, the policy is:
# - stick (0) if the player's sum is 20 or 21
# - hits (1) otherwise
def policy(player_curr_sum:int) -> int:
    return 0 if player_curr_sum >= 20 else 1


# Get state and action from the current observation
def get_state_action(observation:tuple) -> tuple:
    player_curr_sum, dealer_card, _ = observation
    state = (player_curr_sum, dealer_card)
    action = policy(player_curr_sum)
    return (state, action)


def plot_surface(V):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                        figsize=(8, 6))
    ax.set_box_aspect((1, 1, 0.3))
    V = V[12:22, 1:]

    Y = np.arange(V.shape[0])
    X = np.arange(V.shape[1])
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, V, 
                           cmap='Blues',
                           rstride=1,
                           cstride=1,
                           linewidth=0.5,
                           edgecolor='white',
                           alpha=0.9
                           )

    ax.set_xticks(range(0, 11), ['A'] + [f'{i}' for i in range(2, 11)] + ['Face'])
    ax.set_yticks(range(0, 10), range(12, 22))
    ax.set_zticks([-1.0, 0, 1.0])

    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.view_init(elev=33, azim=-50)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.show()



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
    Returns = defaultdict(list)

    # Hyper parameters
    num_episodes = 5_000
    gamma = 1.0

    for _ in range(num_episodes):
        terminated = False
        observation, into = env.reset()
        state, action = get_state_action(observation)

        traj = [] # trajectory of (state, action, reward)
        seen = set() # For first-visit MC method

        # Get a trajectory
        while not terminated:
            observation, reward, terminated, truncated, info = env.step(action)
            traj.append((state, action, reward))
            # Get next state and next action
            state, action = get_state_action(observation)
        
        G = 0.
        while traj:
            state, _, reward = traj.pop()
            G = gamma * G + reward
            if state not in seen:
                seen.add(state)
                Returns[state].append(G)
                V[state] = np.mean(Returns[state])
    
    plot_surface(V)