import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import bandit, get_argmax

SEED = 200
np.random.seed(SEED)

# running the k-armed bandit algorithm on non-stationary environment
def run_bandit_non_stationary(K:int, 
            q_star:np.ndarray,
            rewards:np.ndarray,
            optim_acts_ratio:np.ndarray,
            epsilon:float,
            method:str, # method should be in ["sample", "weighted"] 
            alpha:float=None,
            num_steps:int=1000) -> None:
    
    Q = np.zeros(K) # The action-value for each actions 
    N = np.zeros(K) # The number of times each action been selected    
    ttl_optim_acts = 0

    q_star_temp = np.copy(q_star)

    assert method in ['sample', 'weighted'], "The method should be 'sample' or 'weighted'"

    for i in range(num_steps):
        # print(q_star)
        # get action
        A = None
        if np.random.random() > epsilon:
            A = get_argmax(Q)
        else:
            A = np.random.randint(0, K)
        
        R, is_optim = bandit(q_star_temp, A)

        if method == 'sample':
            N[A] += 1
            Q[A] += (R - Q[A]) / N[A]
        else:
            Q[A] += alpha * (R - Q[A])

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)

        # Updating q_star values
        q_step_scale = 0.01
        q_steps = np.random.normal(loc=0, scale=q_step_scale, size=q_star_temp.shape)
        q_star_temp += q_steps


if __name__ == "__main__":
    
    K = 10
    epsilon = 0.1
    alpha = 0.1

    total_rounds = 2000
    num_steps = 10000
    methods = ['sample', 'weighted']

    q_star = np.zeros(K) # All q starts equally from 0
    rewards = np.zeros(shape=(len(methods), total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(len(methods), total_rounds, num_steps))

    for i, method in enumerate(methods):
        for curr_round in range(total_rounds):
            run_bandit_non_stationary(K, q_star, rewards[i, curr_round], 
                                    optim_acts_ratio[i, curr_round], epsilon,
                                    method=method,
                                    alpha=alpha,
                                    num_steps=num_steps)
    
    rewards = rewards.mean(axis=1)
    optim_acts_ratio = optim_acts_ratio.mean(axis=1)

    plt.plot(rewards, linewidth=1.0)
    plt.show()
    
    record = {
        'hyper_params': methods,
        'rewards': rewards,
        'optim_acts_ratio': optim_acts_ratio,
    }

    # with open('./history/non_stationary_record.pkl', 'wb') as f:
    #     pickle.dump(record, f)

