import numpy as np
import matplotlib.pyplot as plt

SEED = 200
np.random.seed(200)

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
def bandit(q_star, act):
    real_rewards = np.random.normal(q_star, 1.0)
    optim_choice = int(act == np.argmax(real_rewards))
    return real_rewards[act], optim_choice


# running the k-armed bandit algorithm
def run_bandit(K:int, 
            q_star:np.array,
            rewards:np.array,
            optim_acts_ratio: np.array,
            epsilon: float, 
            num_steps:int=1000, 
            curr_round:int=0) -> None:
    
    Q = np.zeros(K) # The average action-value for each actions
    N = np.zeros(K) # The number of times each action been selected    
    optim_acts = 0

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

        optim_acts += is_optim
        rewards[curr_round, i] = R
        optim_acts_ratio[curr_round, i] = optim_acts / (i + 1)



if __name__ == "__main__":

    # Initializing the hyper-parameters
    K = 10 # Number of arms
    epsilon = 0.0
    num_steps = 1000
    total_rounds = 1000

    # Initialize the environment
    q_star = np.random.normal(loc=0, scale=1.0, size=K)
    rewards = np.zeros(shape=(total_rounds, num_steps))
    optim_acts_ratio = np.zeros(shape=(total_rounds, num_steps))
    
    # print(np.random.random() > 0.5)
    # print(np.random.choice(np.arange(10)))
    for curr_round in range(total_rounds):
        # rewards = np.zeros(shape=(total_rounds, num_steps))
        run_bandit(K, q_star, rewards, optim_acts_ratio, epsilon, num_steps, curr_round)
    
    rewards = rewards.mean(axis=0)
    optim_acts_ratio = optim_acts_ratio.mean(axis=0)
    
    plt.plot(optim_acts_ratio)
    plt.show()

    plt.plot(rewards)
    plt.show()