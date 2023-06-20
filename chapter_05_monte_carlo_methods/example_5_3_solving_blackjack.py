import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm

# Initialize the policy
def initialize_policy(shape:tuple) -> np.ndarray:
    policy = np.zeros(shape=shape, dtype=np.int8) # (player_sums, dealer_faces)
    # Following the description of ex. 5.3, the initial policy
    # is to draw when player sums smaller than 20
    policy[:20, :] = 1
    return policy


# Get state and action from the current observation
def get_state_action(observation:tuple, policy:np.ndarray) -> tuple:
    player_curr_sum, dealer_card, _ = observation
    state = (player_curr_sum, dealer_card)
    action = policy[state]
    return state, action


if __name__ == "__main__":

    env = gym.make('Blackjack-v1', sab=True)
    player_sums = env.observation_space[0].n
    dealer_faces = env.observation_space[1].n
    actions = env.action_space.n

    # Initialize action-value function 
    Q = np.zeros(shape=(player_sums, dealer_faces, actions))
    policy = initialize_policy(shape=(player_sums, dealer_faces))
    
    # Here we use incremental update to increase the permance
    Returns = np.zeros_like(Q)
    counts = np.zeros_like(Q)

    # Hyper parameters
    num_episodes = 100_000
    gamma = 1.0

    # Loop over episodes
    for _ in tqdm(range(num_episodes)):
        terminated = False
        observation, info = env.reset()
        
        state, action = get_state_action(observation, policy)
        
        traj = [] # trajectory of (state, action, reward)
        seen = set() # For first-visit MC method

        # Get a trajectory
        while not terminated:
            observation, reward, terminated, truncated, info = env.step(action)
            traj.append((state, action, reward))
            state, action = get_state_action(observation, policy)
        
        G = 0.
        while traj:
            state, action, reward = traj.pop()
            G = gamma * G + reward
            state_action = (*state, action)
            if state_action not in seen:
                seen.add(state_action)
                Returns[state_action] += G
                counts[state_action] += 1
                Q[state_action] = Returns[state_action] / counts[state_action]
                policy[state] = np.argmax(Q[state])

    V = np.max(Q, axis=2)
    utils.plot_surface(V)
        
    # policy = policy[12:22, 1:]
    plt.imshow(policy)
    plt.show()

    