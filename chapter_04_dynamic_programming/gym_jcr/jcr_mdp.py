"""
Still in-progress
"""


import numpy as np
from scipy.stats import poisson


from matplotlib import pyplot as plt
import seaborn as sns
"""
Environment Parameters
"""

REQUEST_RATE = (3., 4.)
RETURN_RATE = (3., 2.)

GAMMA = 0.9
RENTAL_INCOME = 10
TRANSFER_COST = 2
TRANSFER_MAX  = 5
MAX_CAPACITY  = 20

# Location Indicies
A = 0
B = 1

MAX_PMF = 30 # temporary parameter

# action space: negative means transfer from the second location to the first
action_space = np.arange(-TRANSFER_MAX, TRANSFER_MAX+1)


"""
Helper functions
"""

# Construct transition matrix P_{to, from} for one location only for requests
# IMPORTANT: the formate is P_{TO, FROM}
def get_request_transitions_for_one_location(loc):
    assert (loc == A or loc == B)

    transition_matrix = np.zeros(shape=(MAX_CAPACITY+1, MAX_CAPACITY+1))
    # Get poisson distribution
    request_pmf = poisson.pmf(np.arange(MAX_PMF), REQUEST_RATE[loc])
    # Checking if the last pmf is 0
    np.testing.assert_almost_equal(request_pmf[-1], 0., decimal=12)

    for i in range(MAX_CAPACITY + 1):
        for j in range(MAX_CAPACITY + 1):
            if j == 0:
                transition_matrix[i, j] = request_pmf[i:].sum()
            elif j <= i:
                transition_matrix[i, j] = request_pmf[i - j]
    
    return transition_matrix.T


def full_transition_matrix_A(transition_one_loc):
    block_size = MAX_CAPACITY + 1
    transition_matrix = np.zeros(shape=(block_size ** 2, block_size ** 2))
    
    for i in range(block_size):
        transition_matrix[i : block_size ** 2 : block_size,
                          i : block_size ** 2 : block_size] = transition_one_loc
    
    return transition_matrix


def full_transition_matrix_B(transition_one_loc):
    block_size = MAX_CAPACITY + 1
    transition_matrix = np.zeros(shape=(block_size ** 2, block_size ** 2))
    
    for i in range(block_size):
        transition_matrix[i * block_size : (i * block_size + block_size),
                          i * block_size : (i * block_size + block_size)] = transition_one_loc
    
    return transition_matrix


# Construct transition matrix P_{to, from} for one location only for returns
# IMPORTANT: the formate is P_{TO, FROM}
def get_return_transition_matrix_one_location(loc):
    assert (loc == A or loc == B)
    transition_matrix = np.zeros(shape=(MAX_CAPACITY+1, MAX_CAPACITY+1))

    return_pmf = poisson.pmf(np.arange(MAX_PMF), RETURN_RATE[loc])
    np.testing.assert_almost_equal(return_pmf[-1], 0., decimal=12)

    for i in range(MAX_CAPACITY + 1):
        for j in range(MAX_CAPACITY + 1):
            if j == MAX_CAPACITY:
                transition_matrix[i, j] = return_pmf[j-i:].sum()
            elif j >= i and j < MAX_CAPACITY:
                transition_matrix[i, j] = return_pmf[j-i]
    
    return transition_matrix.T

# Moving cars between a and b with an action from action space
def get_moves(a, b, action):
    if action > 0: # from A to B
        return min(a, action)
    else:
        return max(-b, action)

# Get transition matrix of moves at night
def get_nightly_moves():
    pass




if __name__ == '__main__':
    trans_one_loc = get_return_transition_matrix_one_location(0)
    # trans_one_loc = get_request_transitions_for_one_location(0)
    # trans = full_transition_matrix_A(trans_one_loc)
    sns.heatmap(trans_one_loc)
    plt.show()