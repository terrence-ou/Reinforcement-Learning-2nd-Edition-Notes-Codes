import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# take a step, get next state and reward
def env_step(state:tuple,
               act:tuple) -> tuple:

    assert state not in [(0, 0), (3, 3)], "Terminal states should be selected"
    
    next_row = state[0] + act[0]
    next_col = state[1] + act[1]

    if 0 <= next_row < 4 and 0 <= next_col < 4:
        return -1., (next_row, next_col)
    else:
        return -1., state


# Iterative policy evaluation
def policy_evaluation(grid_world:np.array,
                      actions:list,
                      probs:list,
                      gamma:float,
                      delta:float,
                      terminal_states:list) -> tuple:

    # The author uses two-list method in this example
    new_grid_world = np.zeros_like(grid_world)
    
    # Loop over all of the states
    for row in range(4):
        for col in range(4):
            if (row, col) in terminal_states: continue
            curr_state = (row, col)
            for i, act in enumerate(actions):
                reward, next_state = env_step(curr_state, act)
                next_value = grid_world[next_state]
                new_grid_world[curr_state] += probs[i] * (reward + gamma * next_value)
    
    # calculate the abs difference between two grids and return the max diff value
    max_diff = np.abs(new_grid_world - grid_world).max()
    return new_grid_world, max(max_diff, delta)


# plot values and the greedy policy
def plot_value_policy(data:np.ndarray, 
                  axes:np.ndarray,
                  curr_row:int) -> None:
        
        # Plot values
        ax = axes[curr_row, 0]
        sns.heatmap(data, 
                ax=axes[curr_row, 0],
                cbar=False,
                fmt='.1f',
                annot=data,
                cmap='flare',
                linewidths=0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('test')

        # Plot greedy policy
        



if __name__ == '__main__':
    # Initialize the grid world
    grid_world = np.zeros(shape=(4, 4))
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    gamma = 1.0
    theta = 0.001

    # random policy with eqaul probability on each action
    probs = [1. / len(actions)] * len(actions)
    terminal_states = [(0, 0), (3, 3)]

    # store values in the record for plotting
    record = []
    fig, axes = plt.subplots(6, 2, figsize=(3, 10))
    # print(axes)
    curr_row = 0
    # Loop until delta is smaller than theta
    for i in range(1000):

        if i in [0, 1, 2, 3, 10]:
            record.append(grid_world)
            plot_value_policy(grid_world, axes, curr_row)
            curr_row += 1
            

        delta = 0
        grid_world, delta = policy_evaluation(
                            grid_world, 
                            actions, 
                            probs,
                            gamma,
                            delta, 
                            terminal_states)
        
        if delta < theta: break
    
    record.append(grid_world)
    plot_value_policy(grid_world, axes, -1)
    plt.tight_layout()
    plt.show()