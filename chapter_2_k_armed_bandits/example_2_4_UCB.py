import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import get_argmax, bandit

SEED = 200
np.random.seed(SEED)


# running the k-armed bandit algorithm
def run_bandit(K:int, 
            q_star:np.array,
            rewards:np.array,
            optim_acts_ratio: np.array,
            epsilon: float, 
            num_steps:int=1000) -> None:
    
    Q = np.zeros(K)
    N = np.zeros(K) # The number of times each action been selected    
    ttl_optim_acts = 0

    for i in range(num_steps):
        A = None
        # Get action
        if np.random.random() > epsilon:
            A = get_argmax(Q)
        else:
            A = np.random.randint(0, K)
        
        R, is_optim = bandit(q_star, A)
        N[A] += 1
        Q[A] += (R - Q[A]) / N[A]

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)


# running the bandit algorithm with UCB
def run_bandit_UCB(K:int, 
            q_star:np.array,
            rewards:np.array,
            optim_acts_ratio: np.array,
            c: float,
            num_steps:int=1000) -> None:
    
    Q = np.zeros(K)
    N = np.zeros(K) # The number of times each action been selected    
    ttl_optim_acts = 0

    for i in range(num_steps):
        A = None

        # Avoid 0-division:
        # If there's 0 in N, then choose the action with N = 0
        if (0 in N):
            candidates = np.argwhere(N == 0).flatten()
            A = np.random.choice(candidates)
        else:
            confidence = c * np.sqrt(np.log(i) / N)
            freqs = Q + confidence
            A = np.argmax(freqs).flatten()
        
        R, is_optim = bandit(q_star, A)
        N[A] += 1
        Q[A] += (R - Q[A]) / N[A]

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)


if __name__ == "__main__":

    # Initializing the hyper-parameters
    K = 10 # Number of arms
    num_steps = 1000
    total_rounds = 2000
    q_star = np.random.normal(loc=0, scale=1.0, size=K)
    
    hyper_params = {'UCB': 2, 'epsilon': 0.1}

    rewards = np.zeros(shape=(len(hyper_params), total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(len(hyper_params), total_rounds, num_steps))
    
    # Run bandit alg. with e-greedy
    for curr_round in range(total_rounds):
        run_bandit(K, 
                q_star, 
                rewards[0, curr_round], 
                optim_acts_ratio[0, curr_round],
                epsilon=hyper_params['epsilon'],
                num_steps=num_steps)

    # Run UCB and get records
    for curr_round in range(total_rounds):
        run_bandit_UCB(K, 
                q_star, 
                rewards[1, curr_round], 
                optim_acts_ratio[1, curr_round],
                c=hyper_params['UCB'],
                num_steps=num_steps)

    rewards = rewards.mean(axis=1)
    optim_acts_ratio = optim_acts_ratio.mean(axis=1)

    record = {
        'hyper_params': hyper_params, 
        'rewards': rewards,
        'optim_acts_ratio': optim_acts_ratio
    }

    # for val in optim_acts_ratio:
    #     plt.plot(val)
    # plt.show()

    with open('./history/UCB_record.pkl', 'wb') as f:
        pickle.dump(record, f)
