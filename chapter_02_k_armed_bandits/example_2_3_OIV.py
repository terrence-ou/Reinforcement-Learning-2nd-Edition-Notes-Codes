import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import get_argmax, bandit

SEED = 200
np.random.seed(SEED)


# running the k-armed bandit algorithm
def run_bandit(K:int, 
            q_star:np.ndarray,
            rewards:np.ndarray,
            optim_acts_ratio:np.ndarray,
            epsilon:float, 
            num_steps:int=1000,
            init_val:int=0) -> None:
    
    Q = np.ones(K) * init_val # Initial Q values with OIV
    ttl_optim_acts = 0
    alpha = 0.1

    for i in range(num_steps):
        # get action
        A = None
        if np.random.random() > epsilon:
            A = get_argmax(Q)
        else:
            A = np.random.randint(0, K)
        
        R, is_optim = bandit(q_star, A)
        Q[A] += alpha * (R - Q[A])

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)


if __name__ == "__main__":

    # Initializing the hyper-parameters
    K = 10 # Number of arms
    epsilons = [0.1, 0.0]
    init_vals = [0.0, 5.0]
    num_steps = 1000
    total_rounds = 2000

    # Initialize the environment
    q_star = np.random.normal(loc=0, scale=1.0, size=K)
    rewards = np.zeros(shape=(len(epsilons), total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(len(epsilons), total_rounds, num_steps))
    
    # Run the k-armed bandits alg.
    for i, (epsilon, init_val) in enumerate(zip(epsilons, init_vals)):
        for curr_round in range(total_rounds):
            run_bandit(K, q_star, 
                       rewards[i, curr_round], 
                       optim_acts_ratio[i, curr_round], 
                       epsilon=epsilon, 
                       num_steps=num_steps,
                       init_val=init_val)
    
    rewards = rewards.mean(axis=1)
    optim_acts_ratio = optim_acts_ratio.mean(axis=1)

    record = {
        'hyper_params': [epsilons, init_vals], 
        'rewards': rewards,
        'optim_acts_ratio': optim_acts_ratio
    }

    for vals in rewards:
        plt.plot(vals)
    plt.show()
    # with open('./history/OIV_record.pkl', 'wb') as f:
    #     pickle.dump(record, f)
