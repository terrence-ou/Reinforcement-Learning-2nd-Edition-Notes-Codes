import numpy as np


# take a step, get next state and reward
def env_step(state:tuple,
               act:tuple):

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
                      terminal_states:list):

    # The author uses two-list method in this example
    new_grid_world = np.zeros_like(grid_world)
    
    for row in range(4):
        for col in range(4):
            if (row, col) in terminal_states: continue
            curr_state = (row, col)
            for i, act in enumerate(actions):
                reward, next_state = env_step(curr_state, act)
                next_value = grid_world[next_state]
                new_grid_world[curr_state] += probs[i] * (reward + gamma * next_value)
    
    return new_grid_world


if __name__ == '__main__':
    # Initialize the grid world
    grid_world = np.zeros(shape=(4, 4))
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    gamma = 1.0
    theta = 0.001

    # random policy with eqaul probability on each action
    probs = [1. / len(actions)] * len(actions)
    terminal_states = [(0, 0), (3, 3)]

    for i in range(100):
        delta = 0
        grid_world = policy_evaluation(
                            grid_world, 
                            actions, 
                            probs,
                            gamma,
                            delta, 
                            terminal_states)
        if i in [0, 1, 2, 3, 11]:
            print(grid_world, '\n')
    
    print(grid_world)