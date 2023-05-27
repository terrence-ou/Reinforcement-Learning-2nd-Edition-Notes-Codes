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
    Q = np.zeros(shape=(grid_world.shape[0], grid_world.shape[1], len(actions)))

    # Loop over all of the states
    for row in range(4):
        for col in range(4):
            if (row, col) in terminal_states: continue
            for i, act in enumerate(actions):
                reward, next_state = env_step((row, col), act)
                next_value = grid_world[next_state]
                Q[row, col, i] = probs[i] * (reward + gamma * next_value)
    
    # calculate the abs difference between two grids and return the max diff value
    new_grid_world = Q.sum(axis=-1)
    max_diff = np.abs(new_grid_world - grid_world).max()
    return new_grid_world, max(max_diff, delta)


# Get greedy policy from current values
def greedy_policy(grid_world:np.array,
                    actions:list,
                    probs:list,
                    gamma:float,
                    terminal_states:list) -> list:
    
        # The author uses two-list method in this example
    Q = np.zeros(shape=(grid_world.shape[0], grid_world.shape[1], len(actions)))
    act_list = np.array([0x2190, 0x2193, 0x2192, 0x2191])
    greedy_acts = []

    # Loop over all of the states
    for row in range(4):
        acts_row = []
        for col in range(4):
            
            if (row, col) in terminal_states: 
                acts_row.append('')
                continue

            for i, act in enumerate(actions):
                reward, next_state = env_step((row, col), act)
                next_value = grid_world[next_state]
                Q[row, col, i] = probs[i] * (reward + gamma * next_value)
            
            greedy_sels = np.where(np.abs(Q[row, col] - Q[row, col].max()) < 0.0001)[0]
            acts = "".join([chr(act_list[i]) for i in greedy_sels])
            acts_row.append(acts)
        greedy_acts.append(acts_row)

    return greedy_acts


# plot values of the gridworld
def plot_heatmap(data:np.ndarray, 
                annots:list,
                axes:np.ndarray,
                curr_row:int) -> None:
        
        assert len(data.shape) == 2, 'Input must be a 2-D array'
        ax_val = axes[curr_row, 0]
        ax_act = axes[curr_row, 1]
        # Plot values
        sns.heatmap(data, 
                ax=ax_val,
                cbar=False,
                fmt='.1f',
                annot=data,
                cmap='Blues',
                linewidths=0.5)
        ax_val.set_yticks([])
        ax_val.set_xticks([])
        ax_val.set_title(f'k={curr_row}')

        sns.heatmap(data, 
            ax=ax_act,
            cbar=False,
            annot=annots,
            fmt='',
            cmap='Blues',
            linewidths=0.5)
        ax_act.set_yticks([])
        ax_act.set_xticks([])
        ax_act.set_title('test')



if __name__ == '__main__':
    # Initialize the grid world
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    grid_world = np.zeros(shape=(4, 4))
    gamma = 1.0
    theta = 0.001

    # random policy with eqaul probability on each action
    probs = [1. / len(actions)] * len(actions)
    terminal_states = [(0, 0), (3, 3)]

    # store values in the record for plotting
    fig, axes = plt.subplots(6, 2, figsize=(3, 10))
    # print(axes)
    curr_row = 0
    # Loop until delta is smaller than theta
    for i in range(1000):

        if i in [0, 1, 2, 3, 10]:
            greedy_acts = greedy_policy(grid_world, 
                                        actions, 
                                        probs, 
                                        gamma, 
                                        terminal_states)
            plot_heatmap(grid_world, greedy_acts, axes, curr_row)
            curr_row += 1
            

        delta = 0
        grid_world,delta = policy_evaluation(
                            grid_world, 
                            actions, 
                            probs,
                            gamma,
                            delta, 
                            terminal_states)
        
        if delta < theta: break
    

    greedy_acts = greedy_policy(grid_world, 
                                actions, 
                                probs, 
                                gamma, 
                                terminal_states)
    
    plot_heatmap(grid_world, greedy_acts, axes, -1)
    
    plt.tight_layout()
    plt.show()