import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import plot


def plot_result(rewards:np.array, 
                optim_ratio:np.array, 
                legends:list, 
                output_names:list):

    plot(rewards, 
         legends, 
         xlabel='Time step', 
         ylabel='Reward',
         filename=output_names[0]
         )

    # Set tick labels
    fn = lambda: plt.yticks(np.arange(0, 100, 10), labels=[f'{val}%' for val in range(0, 100, 10)])
    plot(optim_ratio, 
         legends, 
         xlabel='Time step', 
         ylabel='% Optimal actions',
         filename=output_names[1],
         fn=fn)


def example_2_2(save_fig=False):
    with open(f'./history/record.pkl', 'rb') as f:
        history = pickle.load(f)

    meta = history['hyper_params']
    rewards = history['rewards']
    optim_ratio = (history['optim_acts_ratio'] * 100)
    
    legends = [f'$\epsilon$={val}' for val in meta]
    
    filenames = ['example_2_2_rewards.png', 'example_2_2_optimal_ratio.png'] if save_fig else [None, None]
    plot_result(rewards, optim_ratio, legends, filenames)


def example_2_3(save_fig=False):
    with open(f'./history/OIV_record.pkl', 'rb') as f:
        history = pickle.load(f)

    meta = history['hyper_params']
    rewards = history['rewards']
    optim_ratio = (history['optim_acts_ratio'] * 100)
    legends = [f'$\epsilon$={meta[0][i]}, $Q_1$={meta[1][i]}' for i in range(len(meta[0]))]
    
    filenames = ['example_2_3_rewards.png', 'example_2_3_optimal_ratio.png'] if save_fig else [None, None]
    plot_result(rewards, optim_ratio, legends, filenames)



def example_2_4(save_fig=False):
    with open(f'./history/UCB_record.pkl', 'rb') as f:
        history = pickle.load(f)

    meta = history['hyper_params']
    rewards = history['rewards']
    optim_ratio = (history['optim_acts_ratio'] * 100)
    legends = [f'$\epsilon$-greedy $\epsilon$={meta["epsilon"]}', f'UCB c={meta["UCB"]}']
    
    filenames = ['example_2_4_rewards.png', 'example_2_4_optimal_ratio.png'] if save_fig else [None, None]
    plot_result(rewards, optim_ratio, legends, filenames)



def exercise_2_5(save_fig=False):
    with open(f'./history/non_stationary_record.pkl', 'rb') as f:
        history = pickle.load(f)

    meta = history['hyper_params']
    rewards = history['rewards']
    optim_ratio = (history['optim_acts_ratio'] * 100)

    legends = [f'{val} method' for val in meta]
    filenames = ['exercise_2_5_rewards.png', 'exercise_2_5_optimal_ratio.png'] if save_fig else [None, None]
    plot_result(rewards, optim_ratio, legends, filenames)


if __name__ == '__main__':

    # example_2_2(save_fig=True)
    example_2_3(save_fig=True)
    # example_2_4(save_fig=False)
    # exercise_2_5(save_fig=True)
