import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Env for this example
def random_move(action:int) -> tuple:
    """
    actions: 0 -> go left; 1: -> go right
    states: 0 -> nonterminal state; 1 -> terminal state
    
    return: (next_state, reward, is_terminated)
    """
    if action == 0:
        prob = np.random.randn()
        if prob <= 0.1:
            return (1, 1, True)
        else:
            return (0, 0, False)
    if action == 1:
        return (1, 0, True)
    

# Get random action
def get_action() -> int:
    return np.random.choice([0, 1], p=[0.5, 0.5])

# Get probability of an action based on the target policy
def prob_target_policy(action: int) -> float:
    return 0. if action == 1 else 1.

# Get probability of an action based on the behavior policy
def prob_behavior_policy() -> float:
    return 0.5 


def plot_result(value_hist:np.ndarray) -> None:
    raise NotImplementedError


def monte_carlo_importance_sampling(total_rounds:int,
                                    episode_per_round:int,
                                    save_record:bool=True) -> np.ndarray:
    
    value_hist = np.zeros(shape=(total_rounds, episode_per_round))
    
    n_episodes = episode_per_round
    n_rounds = total_rounds

    gamma = 1.0

    for r in range(n_rounds):
        V = 0
        count = 0

        for t in tqdm(range(n_episodes)):
            state = 0
            action = get_action()
            terminated = False
            
            traj = []
            seen = set()

            while not terminated:
                next_state, reward, terminated = random_move(action)
                traj.append((state, action, reward))
                state = next_state
                action = get_action()
            
            G = 0
            W = 1.0

            while traj:
                state, action, reward = traj.pop()
                G = G + gamma * reward

                prob_pi = prob_target_policy(action)
                prob_behav = prob_behavior_policy()
                W = W * (prob_pi / prob_behav)

                if state == 0:
                    # seen.add(state)
                    V += W * G
                    count += 1

            value_hist[r, t] = V / count
    
    return value_hist




if __name__ == "__main__":
    total_rounds = 10
    episode_per_round = 100_000

    value_hist = monte_carlo_importance_sampling(total_rounds, episode_per_round)
    
    plt.xscale('log')
    plt.ylim((0, 3.5))
    for i in range(total_rounds):
        plt.plot(value_hist[i])
    
    plt.show()
