import numpy as np


# get a reward based on the given state and action
def get_reward(state:tuple,
               act:tuple):

    if state in [(0, 0), (3, 3)]:
        return 0
    
    next_row = state[0] + act[0]
    next_col = state[1] + act[1]
