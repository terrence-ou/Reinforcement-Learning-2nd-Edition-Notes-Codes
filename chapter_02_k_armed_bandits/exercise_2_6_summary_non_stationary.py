import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
import pickle

from tqdm import tqdm

SEED = 200
np.random.seed(SEED)

######################################################################
#                         HELPER FUNCTIONS                           #
######################################################################

# Get the action with the max Q value
def get_argmax(G:np.ndarray) -> int:
    candidates = np.argwhere(G == G.max()).flatten()
    # return the only index if there's only one max
    if len(candidates) == 1:
        return candidates[0]
    else:
        # instead break the tie randomly
        return np.random.choice(candidates)


# Select arm and get the reward
def bandit(q_star:np.ndarray, 
           act:int) -> tuple:
    real_rewards = np.random.normal(q_star, 1.0)
    # optim_choice = int(real_rewards[act] == real_rewards.max())
    optim_choice = int(q_star[act] == q_star.max())
    return real_rewards[act], optim_choice

# Update policy in gradient bandit algorithm
def update_policy(H:np.ndarray) -> np.ndarray:
    return np.exp(H) / np.exp(H).sum()

# Update preference values for gradient bandit algorithm
def update_H(H:np.ndarray, 
             policy:np.ndarray, 
             alpha:float,
             A:int, 
             curr_reward:float, 
             avg_reward:float) -> np.ndarray:
    selec = np.zeros(len(H), dtype=np.float32)
    selec[A] = 1.0
    H = H + alpha * (curr_reward - avg_reward) * (selec - policy)
    return H


######################################################################
#                            ALGORITHMS                              #
######################################################################

# e_greedy algorithm
def e_greedy_non_stationary(
            K:int,
            q_star:np.ndarray,
            rewards:np.ndarray,
            optim_acts_ratio:np.ndarray,
            epsilon:float,
            alpha:float=None,
            num_steps:int=1000) -> None:
    
    Q = np.zeros(K) # The action-value for each actions 
    ttl_optim_acts = 0

    q_star_temp = np.copy(q_star) # Make a copy of q_star_values

    for i in tqdm(range(num_steps)):
        A = None
        if np.random.random() > epsilon:
            A = get_argmax(Q)
        else:
            A = np.random.randint(0, K)
        
        R, is_optim = bandit(q_star_temp, A)

        Q[A] += alpha * (R - Q[A])

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)

        # Updating q_star values
        q_step_scale = 0.01
        q_steps = np.random.normal(loc=0, scale=q_step_scale, size=q_star_temp.shape)
        q_star_temp += q_steps



# UCB algorithm
def UCB_non_stationary(K:int, 
            q_star:np.ndarray,
            rewards:np.ndarray,
            optim_acts_ratio:np.ndarray,
            c:float,
            alpha:float,
            num_steps:int=1000) -> None:
    
    Q = np.zeros(K)
    N = np.zeros(K) # The number of times each action been selected    
    ttl_optim_acts = 0

    q_star_temp = np.copy(q_star)

    for i in tqdm(range(num_steps)):
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
        
        R, is_optim = bandit(q_star_temp, A)
        N[A] += 1
        Q[A] += alpha * (R - Q[A])

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)

        # Updating q_star values
        q_step_scale = 0.01
        q_steps = np.random.normal(loc=0, scale=q_step_scale, size=q_star_temp.shape)
        q_star_temp += q_steps


# gradient bandit algorithm
def gradient_bandit_non_stationary(K:int, 
            q_star:np.ndarray,
            rewards:np.ndarray,
            optim_acts_ratio:np.ndarray,
            alpha:float,
            baseline:bool,
            num_steps:int=1000) -> None:

    q_star_temp = np.copy(q_star)
    H = np.zeros(K, dtype=np.float32) # initialize preference 
    policy = np.ones(K, dtype=np.float32) / K
    ttl_reward = 0
    ttl_optim_acts = 0
    
    for i in tqdm(range(num_steps)):

        A = np.random.choice(np.arange(K), p=policy)
        reward, is_optim = bandit(q_star_temp, A)
        avg_reward = 0

        if baseline:
            # Get average reward unitl timestep=i
            avg_reward = ttl_reward / i if i > 0 else reward

        # Update preference and policy
        H = update_H(H, policy, alpha, A, reward, avg_reward)
        policy = update_policy(H)

        ttl_reward += reward
        ttl_optim_acts += is_optim
        rewards[i] = reward
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)

        # Updating q_star values
        q_step_scale = 0.01
        q_steps = np.random.normal(loc=0, scale=q_step_scale, size=q_star_temp.shape)
        q_star_temp += q_steps


# Optimistic Initial Value
def OIV_non_stationary(K:int, 
            q_star:np.ndarray,
            rewards:np.ndarray,
            optim_acts_ratio:np.ndarray,
            epsilon:float, 
            alpha:float,
            num_steps:int=1000,
            init_val:int=0) -> None:
    
    Q = np.ones(K) * init_val # Initial Q values with OIV
    ttl_optim_acts = 0

    q_star_temp = np.copy(q_star)
    
    for i in tqdm(range(num_steps)):
        # get action
        A = None
        if np.random.random() > epsilon:
            A = get_argmax(Q)
        else:
            A = np.random.randint(0, K)
        
        R, is_optim = bandit(q_star_temp, A)
        Q[A] += alpha * (R - Q[A])

        ttl_optim_acts += is_optim
        rewards[i] = R
        optim_acts_ratio[i] = ttl_optim_acts / (i + 1)

        # Updating q_star values
        q_step_scale = 0.01
        q_steps = np.random.normal(loc=0, scale=q_step_scale, size=q_star_temp.shape)
        q_star_temp += q_steps


######################################################################
#                         Wrapper Function                           #
######################################################################

# A wraper function for running differen algorithms
def run_algorithm(fn_name:str,
                    fn:'function',
                    params:np.ndarray,
                    args:dict,
                    total_rounds:int) -> np.ndarray:

    if fn_name == 'e_greedy':
        hyper_param = 'epsilon'
    elif fn_name == 'ucb':
        hyper_param = 'c'
    elif fn_name == 'gradient':
        hyper_param = 'alpha'
    elif fn_name == 'oiv':
        hyper_param = 'init_val'

    args[hyper_param] = None

    rewards_hist = np.zeros(shape=(len(params), total_rounds, args['num_steps']))
    optm_acts_hist = np.zeros_like(rewards_hist)
    for i, param in enumerate(params):
        args[hyper_param] = param
        for curr_round in range(total_rounds):
            print('current round: ', curr_round)
            fn(**args, 
               rewards=rewards_hist[i, curr_round],
               optim_acts_ratio=optm_acts_hist[i, curr_round])
    
    # get last 100_000 rewards
    rewards_hist = rewards_hist[:, :, -100_000:]
    return rewards_hist.mean(axis=1).mean(axis=1)


if __name__ == "__main__":
    K = 10
    num_steps = 200_000
    total_rounds = 10 # it takes around 20 minutes to run
    q_star = np.random.normal(loc=0, scale=1.0, size=K)

    # Creating parameter array: [1/128, 1/64, 1/32, 1/16, ...]
    multiplier = np.exp2(np.arange(10))
    params = np.ones(10) * (1 / 128)
    params *= multiplier
    x_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']

    # Creating a dict to keep track on running records
    records = {'params': params,
               'x_labels': x_labels}
    history = namedtuple('history', ['bounds', 'data'])

    base_args = {
        'K': K,
        'q_star': q_star,
        'num_steps': num_steps
    }


    # ======== e_greedy ========
    print("======== Running epsilon_greedy Algorithm ========")
    eps_bounds = [0, 6]
    fn_params = params[eps_bounds[0]: eps_bounds[1]]

    eps_args = base_args.copy()
    eps_args['alpha'] = 0.1

    eps_rewards = run_algorithm('e_greedy', e_greedy_non_stationary, fn_params, eps_args, total_rounds)
    records['e_greedy'] = history(eps_bounds, eps_rewards)
    # plt.plot(np.arange(eps_bounds[0], eps_bounds[1]), eps_rewards)


    # ======== UCB ========
    print("======== Running Upper Confidence Bound Algorithm ========")
    ucb_bounds = [3, 10]
    fn_params = params[ucb_bounds[0]: ucb_bounds[1]]

    ucb_args = base_args.copy()
    ucb_args['alpha'] = 0.1
    
    ucb_rewards = run_algorithm('ucb', UCB_non_stationary, fn_params, ucb_args, total_rounds)
    records['ucb'] = history(ucb_bounds, ucb_rewards)
    # plt.plot(np.arange(ucb_bounds[0], ucb_bounds[1]), ucb_rewards)


    # ======== Gradient ========
    print("======== Running Gradient Bandit Algorithm ========")
    gd_bounds = [2, 9]
    fn_params = params[gd_bounds[0]:gd_bounds[1]]
    gd_args = base_args.copy()
    gd_args['baseline'] = True

    gd_rewards = run_algorithm('gradient', gradient_bandit_non_stationary, fn_params, gd_args, total_rounds)
    records['gradient'] = history(gd_bounds, gd_rewards)

    # plt.plot(np.arange(gd_bounds[0], gd_bounds[1]), gd_rewards)


    # ======== OIV ========
    print("======== Running Optimisitc Initial Value Algorithm ========")
    oiv_bounds = [5, 10]
    fn_params = params[oiv_bounds[0]:oiv_bounds[1]]
    oiv_args = base_args.copy()
    oiv_args['epsilon'] = 0.1
    oiv_args['alpha'] = 0.1
    oiv_rewards = run_algorithm('oiv', OIV_non_stationary, fn_params, oiv_args, total_rounds)
    records['oiv'] = history(oiv_bounds, oiv_rewards)
    plt.plot(np.arange(oiv_bounds[0], oiv_bounds[1]), oiv_rewards)

    # save histories
    with open('./history/exercise_2_6.pkl', 'wb') as f:
        pickle.dump(records, f)

    # plt.show()