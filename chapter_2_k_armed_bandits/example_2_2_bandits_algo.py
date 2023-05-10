import numpy as np
import matplotlib.pyplot as plt
import pickle


SEED = 200
np.random.seed(SEED)

# Get the action with the max Q value
def get_argmax(G:np.array) -> int:
    candidates = np.argwhere(G == G.max()).flatten()
    # return the only index if there's only one max
    if len(candidates) == 1:
        return candidates[0]
    else:
        # instead break the tie randomly
        return np.random.choice(candidates)


# select arm and get the reward
def bandit(q_star:np.array, 
           act:int) -> tuple:
    real_rewards = np.random.normal(q_star, 1.0)
    optim_choice = int(act == np.argmax(real_rewards))
    return real_rewards[act], optim_choice


# running the k-armed bandit algorithm
def run_bandit(K:int, 
            q_star:np.array,
            rewards:np.array,
            optim_acts_ratio: np.array,
            epsilon: float, 
            num_steps:int=1000) -> None:
    
    Q = np.zeros(K) # The average action-value for each actions
    N = np.zeros(K) # The number of times each action been selected    
    ttl_optim_acts = 0

    for i in range(num_steps):
        # print(q_star)
        # get action
        A = None
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


if __name__ == "__main__":

    # Initializing the hyper-parameters
    K = 10 # Number of arms
    epsilons = [0.0, 0.01, 0.1]
    num_steps = 1000
    total_rounds = 10

    # Initialize the environment
    q_star = np.random.normal(loc=0, scale=1.0, size=K)
    rewards = np.zeros(shape=(len(epsilons), total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(len(epsilons), total_rounds, num_steps))
    
    # Run the k-armed bandits alg.
    for i, epsilon in enumerate(epsilons):
        for curr_round in range(total_rounds):
            run_bandit(K, q_star, rewards[i, curr_round], optim_acts_ratio[i, curr_round], epsilon, num_steps)
    
    rewards = rewards.mean(axis=1)
    optim_acts_ratio = optim_acts_ratio.mean(axis=1)

    record = {
        'epsilons': epsilons, 
        'rewards': rewards,
        'optim_ratio': optim_acts_ratio
    }

    with open('./history/record.pkl', 'wb') as f:
        pickle.dump(record, f)
