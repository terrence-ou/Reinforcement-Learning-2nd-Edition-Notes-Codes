import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from race_track_env.race_track import RaceTrack

# soft behavior policy
def behavior_pi(state:tuple, 
                nA:int, 
                target_pi:np.ndarray, 
                epsilon:float) -> tuple:
    '''
    The behaviro policy returns both the action and 
    the probability of that action
    '''

    rand_val = np.random.rand()
    greedy_act = target_pi[state]
    
    if rand_val > epsilon:
        return greedy_act, (1 - epsilon + epsilon / nA)
    else:
        action = np.random.choice(nA)
        if action == greedy_act:
            return action, (1 - epsilon + epsilon / nA)
        else:
            return action, epsilon / nA


# Plot the result
def plot_result(value_hist:dict, total_episodes) -> None:
    
    line_width = 1.2
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}

    plt.figure(figsize=(10, 6), dpi=150)
    plt.ylim((-500.0, 0.0))
    plt.grid(c='lightgray')
    plt.margins(0.02)

    # Draw/remove axis lines
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
    
    x = np.arange(total_episodes)
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000, 10_000, 100_000, 1_000_000], 
               ['1', '10', '100', '1000', '10,000', '100,000', '1,000,000'])

    colors = ['tomato', 'cornflowerblue']
    for i, (key, value) in enumerate(value_hist.items()):
        title, label = key.split(',')
        plt.plot(x, uniform_filter(value, size=20), 
                 linewidth=line_width, 
                 label=label,
                 c=colors[i],
                 alpha=0.95)

    plt.title(title + ' training record', fontdict=fontdict)
    plt.xlabel('Episodes (log scale)', fontdict=fontdict)
    plt.ylabel('Rewards', fontdict=fontdict)    
    plt.legend()
    plt.savefig(f'./plots/exercise_5_12/{"_".join(title.lower().split())}.png')
    plt.show()


# Off-policy monte carlo importance sampling algorithm
def off_policy_monte_carlo(total_episodes:int,
                            track_map:str, render_mode:str,
                            zero_acc:bool=False) -> tuple:

    gamma = 0.9
    epsilon = 0.1

    env = RaceTrack(track_map, render_mode, size=20)
    action_space = env.nA # (9, ), nine actions in total
    observation_space = env.nS # (curr_row, curr_col, row_speed, col_speed)

    Q = np.random.normal(size=(*observation_space, action_space))
    Q -= 500 # optimistic initial values
    C = np.zeros_like(Q)
    target_pi = np.argmax(Q, axis=-1)

    reward_hist = np.zeros(shape=(total_episodes), dtype=np.float32)

    for i in range(total_episodes):
        trajectory = []
        terminated = False
        state, info = env.reset()
        (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)
        
        ttl_reward = 0

        # Generate a trajectory using behaviro policy
        while not terminated:
            if zero_acc and np.random.rand() <= 0.1:
                non_acc_act = 4 # check env._action_to_acceleration
                observation, reward, terminated = env.step(non_acc_act)
            else:
                observation, reward, terminated = env.step(action)
            
            ttl_reward += reward
            trajectory.append((state, action, reward, act_prob))
            state = observation
            (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)
        
        G = 0.
        W = 1.
        # Loop inversely to update G and Q values
        while trajectory:
            (state, action, reward, act_prob) = trajectory.pop()
            G = gamma * G + reward
            C[state][action] = C[state][action] + W
            Q[state][action] = Q[state][action] + (W / C[state][action]) * (G - Q[state][action])

            target_pi[state] = np.argmax(Q[state])
            if action != target_pi[state]:
                break
            W = W * (1 / act_prob)
        
        reward_hist[i] = ttl_reward
        if i % 1000 == 0:
            print(f'Episode: {i}, reward: {ttl_reward}, epsilon: {epsilon}')
    return reward_hist, Q



if __name__ == "__main__":

    train = False # Switch between train and evaluation
    track_sel = 'a'
    total_episodes = 1_000_000

    if train:

        reward_hist_dict = dict()
        Q_dict = dict()

        for i in range(2):
            track_name = f'Track {track_sel.capitalize()}'
            use_zero_acc = 'with zero acc.' if i else 'without zero acc.'
            key = track_name + ',' + use_zero_acc

            reward_hist, Q = off_policy_monte_carlo(total_episodes, track_sel, None, i)
            reward_hist_dict[key] = reward_hist
            Q_dict[key] = Q
        
        plot_result(reward_hist_dict, total_episodes)
        with open(f'./history/exercise_5_12/track_{track_sel}.pkl', 'wb') as f:
            pickle.dump(Q_dict, f)

    else: # Evaluate the Q values and plot sample paths

        with open(f'./history/exercise_5_12/track_{track_sel}.pkl', 'rb') as f:
            Q_dict = pickle.load(f)

        key = list(Q_dict.keys())[0]
        Q = Q_dict[key]
        policy = np.argmax(Q, axis=-1) # greedy policy
        
        env = RaceTrack(track_sel, None, 20)
        fig = plt.figure(figsize=(12, 5), dpi=150)
        fig.suptitle('Sample trajectories', size=12, weight='bold')

        for i in range(10):
            track_map = np.copy(env.track_map)
            state, obs = env.reset()
            terminated = False
            while not terminated:
                track_map[state[0], state[1]] = 0.6 
                action = policy[state]
                next_state, reward, terminated = env.step(action)
                state = next_state

            ax = plt.subplot(2, 5, i + 1)
            ax.axis('off')
            ax.imshow(track_map, cmap='GnBu')
        plt.tight_layout()
        plt.savefig(f'./plots/exercise_5_12/track_{track_sel}_paths.png')
        plt.show()
