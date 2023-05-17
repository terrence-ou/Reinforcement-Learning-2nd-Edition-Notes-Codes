import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

from utils import bandit

SEED = 200
np.random.seed(SEED)

def update_policy(H:np.array) -> np.array:
    return np.exp(H) / np.exp(H).sum()

def update_H(H:np.array, 
                policy:np.array, 
                alpha:float,
                A:int, 
                curr_reward:float, 
                avg_reward:float) -> np.array:
    selec = np.zeros(len(H), dtype=np.float32)
    selec[A] = 1.0
    H = H + alpha * (curr_reward - avg_reward) * (selec - policy)
    return H

# running the k-armed bandit algorithm
def run_bandit(K:int, 
            q_star:np.array,
            rewards:np.array,
            optim_acts_ratio: np.array,
            alpha: float,
            num_steps:int=1000) -> None:
    
    H = np.zeros(K, dtype=np.float32) # initialize preference 
    policy = np.ones(K, dtype=np.float32) / K
    ttl_reward = 0
    ttl_optim_acts = 0
    
    for i in range(num_steps):

        A = np.random.choice(K, p=policy)
        reward, is_optim = bandit(q_star, A)
        # Get average reward unitl timestep=i
        avg_reward = ttl_reward / i if i > 0 else reward
        # Update preference and policy
        H = update_H(H, policy, alpha, A, reward, avg_reward)
        policy = update_policy(H)

        ttl_reward += reward
        ttl_optim_acts += is_optim
        rewards[i] = reward
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)


if __name__ == "__main__":

    # Initializing the hyper-parameters
    K = 10 # Number of arms
    alpha = [0.1, 0.4]
    baseline = [0.0, 4.0]
    hyper_params = list(itertools.product(baseline, alpha))
    
    num_steps = 1000
    total_rounds = 2000

    rewards = np.zeros(shape=(len(hyper_params), total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(len(hyper_params), total_rounds, num_steps))

    for i, (b, a) in enumerate(hyper_params):
        # Initialize the environment
        q_star = np.random.normal(loc=b, scale=1.0, size=K)
        for curr_round in range(total_rounds):
            run_bandit(K, 
                    q_star, 
                    rewards[i, curr_round], 
                    optim_acts_ratio[i, curr_round],
                    a,
                    num_steps)
    
    optim_acts_ratio = optim_acts_ratio.mean(axis=1)

    # for val in optim_acts_ratio:
    #     plt.plot(val)
    # plt.show()
    
    record = {
        'hyper_params': hyper_params, 
        'optim_acts_ratio': optim_acts_ratio
    }

    with open('./history/bandit_record.pkl', 'wb') as f:
        pickle.dump(record, f)


