import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# get a reward based on the state and action
def get_reward(state, act):
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
def value_update(grid_world, actions):
    '''
    grid_world: the 5x5 grid world map, numpy array
    '''
    # Hex code for (right, down, left, up)
    act_lists = np.array([0x2190, 0x2193, 0x2192, 0x2191])
    optim_acts = []
    for row in range(5):
        acts = []
        for col in range(5):
            value_candidates = np.zeros(shape=(len(actions)))
            # Iterate over all four actions
            for i , act in enumerate(actions):
                reward, next_state = get_reward((row, col), act)
                next_value = grid_world[next_state]
                # Get a discounted next value for current action
                value_candidates[i] = reward + gamma * next_value
            # Bellman optimal equation
            grid_world[row, col] = value_candidates.max()

            # Finding the optimal action(s)
            max_args = np.where(value_candidates == value_candidates.max())
            selected_acts = ''.join([chr(c) for 
                                        c in act_lists[max_args].tolist()])
            acts.append(selected_acts)

        optim_acts.append(acts)

    return optim_acts


# plot the value table
def plot_grid(grid_world, annot = None):
    annot_kwargs = {
        'fontsize': '18',
        # 'fontweight': 'bold'
    }
    fmt = ''
    file_name = 'optimal_actions'
    
    if annot == None:
        annot = True
        fmt = '.1f'
        file_name = 'optimal_values'
    
    plt.figure(figsize=(6, 6), dpi=100)
    sns.heatmap(grid_world, 
                annot=annot,
                annot_kws=annot_kwargs,
                linewidths=1., 
                # fmt='.1f',
                fmt=fmt,
                cmap='crest',
                cbar=False)
    plt.tick_params(
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False)
    plt.title(f"{' '.join(file_name.split('_')).capitalize()}", fontsize=20, fontweight='bold', pad=10)
    plt.savefig(f'./plots/example_3_8/{file_name}.png')
    # plt.show()


if __name__ == "__main__":
    # Initialize the grid world
    grid_world = np.zeros(shape=(5, 5))
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    gamma = 0.9

    for i in range(40):
        prev_grid = np.copy(grid_world)
        optim_acts = value_update(grid_world, actions)
        # if i % 2 == 0:
        #     plot_grid(grid_world, i)
        if np.allclose(prev_grid, grid_world, rtol=0.001):
            print('done', i)
            # print(optim_acts)
            plot_grid(grid_world, annot = optim_acts)
            plot_grid(grid_world, annot = None)
            break
