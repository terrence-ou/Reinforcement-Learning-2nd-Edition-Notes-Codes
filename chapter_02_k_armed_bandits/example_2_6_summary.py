import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
import pickle

from example_2_2_bandits_algo import run_bandit as e_greedy
from example_2_3_OIV import run_bandit as OIV
from example_2_4_UCB import run_bandit_UCB as UCB
from example_2_5_gradient import run_bandit as gradient


SEED = 200
np.random.seed(SEED)

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
            fn(**args, 
               rewards=rewards_hist[i, curr_round],
               optim_acts_ratio=optm_acts_hist[i, curr_round])
    
    return rewards_hist.mean(axis=1).mean(axis=1)


if __name__ == "__main__":
    K = 10
    num_steps = 1000
    total_rounds = 2000
    q_star = np.random.normal(loc=0, scale=1.0, size=K)

    # Creating parameter array: [1/128, 1/64, 1/32, 1/16, ...]
    multiplier = np.exp2(np.arange(10))
    params = np.ones(10) * (1 / 128)
    params *= multiplier
    x_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']

    # Creating a dict to record running histories
    records = {'params': params,
               'x_labels': x_labels}
    history = namedtuple('history', ['bounds', 'data'])

    base_args = {
        'K': K,
        'q_star': q_star,
        'num_steps': num_steps
    }

    # ======== e_greedy ========
    eps_bounds = [0, 6]
    fn_params = params[eps_bounds[0]: eps_bounds[1]]

    eps_rewards = run_algorithm('e_greedy', e_greedy, fn_params, base_args.copy(), total_rounds)
    records['e_greedy'] = history(eps_bounds, eps_rewards)


    # ======== UCB ========
    ucb_bounds = [3, 10]
    fn_params = params[ucb_bounds[0]: ucb_bounds[1]]

    ucb_rewards = run_algorithm('ucb', UCB, fn_params, base_args.copy(), total_rounds)
    records['ucb'] = history(ucb_bounds, ucb_rewards)


    # ======== Gradient ========
    gd_bounds = [2, 9]
    fn_params = params[gd_bounds[0]:gd_bounds[1]]
    gd_args = base_args.copy()
    gd_args['baseline'] = True

    gd_rewards = run_algorithm('gradient', gradient, fn_params, gd_args, total_rounds)
    records['gradient'] = history(gd_bounds, gd_rewards)


    # ======== OIV ========
    oiv_bounds = [5, 10]
    fn_params = params[oiv_bounds[0]:oiv_bounds[1]]
    oiv_args = base_args.copy()
    oiv_args['epsilon'] = 0.0
    oiv_rewards = run_algorithm('oiv', OIV, fn_params, oiv_args, total_rounds)
    records['oiv'] = history(oiv_bounds, oiv_rewards)


    with open('./history/summary.pkl', 'wb') as f:
        pickle.dump(records, f)