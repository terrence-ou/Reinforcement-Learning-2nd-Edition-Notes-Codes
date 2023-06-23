import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm

# Initialize the policy
def initialize_policy(shape:tuple) -> tuple:
    policy = np.zeros(shape=shape, dtype=np.int8) # (player_sums, dealer_faces)
    # Following the description of ex. 5.3, the initial policy
    # is to draw when player sums smaller than 20
    policy[:20, :] = 1
    return policy


# Monte-Carlo ES
def monte_carlo_es(num_episodes=10_000, save_record=False) -> np.ndarray:

    env = gym.make('Blackjack-v1', sab=True)
    player_sums = env.observation_space[0].n
    dealer_faces = env.observation_space[1].n
    aces = env.observation_space[2].n
    actions = env.action_space.n

    # Initialize action-value function 
    Q = np.zeros(shape=(player_sums, dealer_faces, aces, actions))
    policy = initialize_policy(shape=(player_sums, dealer_faces, aces))
    
    # Here we use incremental update to increase the permance
    Returns = np.zeros_like(Q)
    counts = np.zeros_like(Q)

    # Hyper parameters
    gamma = 1.0

    # Loop over episodes
    for _ in tqdm(range(num_episodes)):
        terminated = False
        
        observation, info = env.reset()
        # Exploring start
        state = observation
        action = np.random.randint(0, 2) 
        
        traj = [] # trajectory of (state, action, reward)
        seen = set() # For first-visit MC method

        # Get a trajectory
        while not terminated:
            observation, reward, terminated, truncated, info = env.step(action)
            traj.append((state, action, reward))
            state = observation
            action = policy[state]

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
        
        if save_record:
            with open('./history/example_5_3.pkl', 'wb') as f:
                record = {'Q': Q,
                        'policy': policy}
                pickle.dump(record, f)

    return Q, policy



if __name__ == "__main__":

    num_episodes = 1_000_000

    # Q, policy = monte_carlo_es(num_episodes, save_record=False)

    with open('./history/example_5_3.pkl', 'rb') as f:
        record = pickle.load(f)
        Q = record['Q']
        policy = record['policy']

    V = np.max(Q, axis=-1)
    utils.plot_surface(V[:, :, 1])
    policy = policy[12:22, 1:, 1]
    plt.imshow(policy)
    plt.show()

    