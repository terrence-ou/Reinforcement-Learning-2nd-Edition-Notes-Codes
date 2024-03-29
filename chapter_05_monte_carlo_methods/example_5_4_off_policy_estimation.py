import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from typing import Any
from tqdm import tqdm
import pickle


# Get probability of an action given a state based on the target policy
def prob_target_policy(player_sum:int, action:int) -> float:
    if player_sum >= 20 and action == 0:
        return 1.
    if player_sum < 20 and action == 1:
        return 1.
    return 0.


# Get probability of an action given a state based on the behaviro policy
def prob_behavior_policy() -> float:
    return 0.5 # as defined in the book


# get random action
def get_action() -> int:
    return np.random.choice([0, 1], p=[0.5, 0.5])


# Calculate MSE loss between given value and target value
def mse(val_1:float, val_2:float) -> float:
    return np.sqrt((val_1 - val_2) ** 2)


# initialize the state to the preset start state
def init_start_state(env:Any, start_state:tuple) -> tuple:
    observation, info = env.reset()
    while observation != start_state:
        observation, info = env.reset()
    return env, observation


# plotting function
def plot_result(value_hist:dict) -> None:

    line_width = 1.5
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}

    ord_hist = value_hist['ordinary'].mean(axis=0)
    weighted_hist = value_hist['weighted'].mean(axis=0)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.ylim((0.0, 2.0))
    plt.grid(c='lightgray')
    plt.margins(0.02)

    # get rid of the top/right frame lines
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]: 
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)

    x = np.arange(len(ord_hist))
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1000', '10,000'])

    plt.plot(x, weighted_hist, linewidth=line_width, c='tomato',
             label='Weighted importance sampling')
    plt.plot(x, ord_hist, linewidth=line_width, c='lightseagreen',
             label='Ordinary importance sampling')
    
    plt.xlabel('Episodes (log scale)', fontdict=fontdict)
    plt.ylabel('Mean square error\n(avg. over 20 runs)', fontdict=fontdict)

    plt.legend(loc=7)
    # plt.show()
    plt.savefig('./plots/example_5_4.png')



# run monte carlo off-policy importance sampling 
def monte_carlo_importance_sampling(total_rounds:int, 
                                    episodes_per_round:int, 
                                    save_record:bool=True) -> dict:
    start_state = (13, 2, 1) # player sum = 13, dealer face = 2, usable ace = 1
    target_val = -0.27726 # Provided in the book
    total_episodes = episodes_per_round
    gamma = 1.0
    n_rounds = total_rounds

    value_hist = {'ordinary': np.zeros(shape=(n_rounds, total_episodes)),
                  'weighted': np.zeros(shape=(n_rounds, total_episodes))}

    env = gym.make('Blackjack-v1', sab=True)

    for r in range(n_rounds):
        print(f'Running {r:3d} round sampling')
        # Initialize episode-wise variables
        V_ord = 0 # Values of the ordinary sampling
        V_wei = 0 # Values of the weighted sampling
        rho = 0 # cumulative sum
        count = 0 # the number of the state been visited

        # Epsode
        for t in tqdm(range(total_episodes)):
            
            terminated = False
            env, observation = init_start_state(env, start_state)
            state = observation # for uniform naming purpose
            action = get_action()
            
            # Get a trajectory
            traj = []
            while not terminated:
                observation, reward, terminated, truncated, info = env.step(action)
                traj.append((state, action, reward))
                state = observation
                action = get_action()
            
            G = 0 # Return value
            W = 1.0 # The importance sampling ratio

            while traj:                
                state, action, reward = traj.pop()
                G = gamma * G + reward
                # update the importance sampling ratio
                prob_pi = prob_target_policy(state[0], action)
                prob_behav = prob_behavior_policy()
                W = W * (prob_pi / prob_behav)

                if state == start_state:                    
                    # Update ordinary sampling value
                    V_ord += W * G
                    count += 1
                    # Update weighted sampling value
                    V_wei += W * G
                    rho = rho + W

            # Get MSE loss between current values and the target value
            mse_wei = mse(V_wei / rho if rho != 0 else 0, target_val)
            mse_ord = mse(V_ord / count, target_val)
            value_hist['weighted'][r, t] = mse_wei
            value_hist['ordinary'][r, t] = mse_ord
    
    if save_record:
        with open('./history/example_5_4.pkl', 'wb') as f:
            pickle.dump(value_hist, f)
    return value_hist



if __name__ ==  "__main__":

    total_rounds = 20
    episodes_per_round = 10_000

    value_hist = monte_carlo_importance_sampling(total_rounds, episodes_per_round, save_record=True)

    # with open('./history/example_5_4.pkl', 'rb') as f:
        # value_hist = pickle.load(f)

    plot_result(value_hist)