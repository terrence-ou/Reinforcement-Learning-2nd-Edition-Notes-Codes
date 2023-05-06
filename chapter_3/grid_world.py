import numpy as np

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
def value_update(grid_world):
    for row in range(5):
        for col in range(5):
            curr_value = 0
            for act in actions:
                reward, next_state = get_reward((row, col), act)
                next_value = grid_world[next_state]
                curr_value += 0.25 * (reward + gamma * next_value)
            grid_world[row, col] = curr_value



if __name__ == "__main__":
    # Initialize the grid world
    grid_world = np.zeros(shape=(5, 5))
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    gamma = 0.9
    for i in range(100):
        prev_grid = np.copy(grid_world)
        value_update(grid_world)
        if np.allclose(prev_grid, grid_world, rtol=0.001):
            print('done', i)
            break

    print(np.around(grid_world, decimals=1))