import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot(data, legends, xlabel, ylabel, fn=lambda:None, filename=None):
    fontdict={
        'fontsize': 12,
        'fontweight': 'bold',
    }

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(c='lightgray')
    plt.margins(0.02)
    # revers the loop for a better visualization
    for i in range(len(data)-1, -1, -1):
        plt.plot(data[i], label=f"$\epsilon$={legends[i]}", linewidth=1.5)

        # get rid of the frame
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]: 
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)

    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel(ylabel, fontdict=fontdict)
    plt.legend(loc=4, fontsize=13)
    fn()

    if not filename:
        plt.show()
    else:
        plt.savefig(f'./plots/{filename}')


if __name__ == '__main__':
    with open('./history/record.pkl', 'rb') as f:
        history = pickle.load(f)
    epsilons = history['epsilons']
    rewards = history['rewards']
    optim_ratio = (history['optim_ratio'] * 100)

    plot(rewards, epsilons, 
         xlabel='Time step', 
         ylabel='Reward',
         filename='example_2_2_rewards.png')

    # Set tick labels
    fn = lambda: plt.yticks(np.arange(0, 100, 10), labels=[f'{val}%' for val in range(0, 100, 10)])
    plot(optim_ratio, epsilons, 
         xlabel='Time step', 
         ylabel='% Optimal actions',
         fn=fn,
         filename='example_2_2_optimal_ratio.png')