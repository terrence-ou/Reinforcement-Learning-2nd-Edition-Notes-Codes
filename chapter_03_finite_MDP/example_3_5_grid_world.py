import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# get a reward based on the state and action
def get_reward(state:tuple, act: tuple):
    '''
    state: current state, tuple (row, col)
    act: action, tuple (d_row, d_col)
    
    return: reward and next state
    '''
    if state == (0, 1):
        return 10, (4, 1)
    if state == (0, 3):
        return 5, (2, 3)
    
    next_row = state[0] + act[0]
    next_col = state[1] + act[1]

    if not (0 <= next_row < 5 and 0 <= next_col < 5):
        return -1, state

    return 0, (next_row, next_col)


# value function
def value_update(grid_world:np.array):
    '''
    grid_world: the 5x5 grid world map, numpy array
    '''
    for row in range(5):
        for col in range(5):
            curr_value = 0
            for act in actions:
                reward, next_state = get_reward((row, col), act)
                next_value = grid_world[next_state]
                curr_value += 0.25 * (reward + gamma * next_value)
            grid_world[row, col] = curr_value


# plot the value table
def plot_grid(grid_world:np.array, iteration:int=0):
    annot_kwargs = {
        'fontsize': '18'
    }
    plt.figure(figsize=(6, 6), dpi=100)
    sns.heatmap(grid_world, 
                annot=True,
                annot_kws=annot_kwargs,
                linewidths=1., 
                fmt='.1f',
                cmap='crest',
                cbar=False)
    plt.tick_params(
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False)
    plt.title(f'Iteration {iteration}', fontsize=20, fontweight='bold', pad=10)
    plt.savefig(f'./plots/example_3_5/{iteration}.png')
    # plt.show()


if __name__ == "__main__":
    # Initialize the grid world
    grid_world = np.zeros(shape=(5, 5))
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    gamma = 0.9

    iter_to_save = {0, 1, 4, 10, 32}

    for i in range(40):
        prev_grid = np.copy(grid_world)
        value_update(grid_world)
        if i in iter_to_save:
            plot_grid(grid_world, i)
        if np.allclose(prev_grid, grid_world, rtol=0.01):
            print('done', i)
            plot_grid(grid_world, i)
            break
