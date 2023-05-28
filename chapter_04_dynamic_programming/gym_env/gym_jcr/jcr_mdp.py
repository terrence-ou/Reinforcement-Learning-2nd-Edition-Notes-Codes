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
    full_cases = (MAX_CAPACITY + 1) ** 2
    transition_matrix = np.zeros(shape=(full_cases, full_cases, action_space.shape[0]))
    for a in range(MAX_CAPACITY + 1):
        for b in range(MAX_CAPACITY + 1):
            for i, action in enumerate(action_space):
                old_state_index = a * (MAX_CAPACITY + 1) + b
                moves = get_moves(a, b, action)
                new_a = min(a - moves, MAX_CAPACITY)
                new_b = min(b + moves, MAX_CAPACITY)
                new_state_index = new_a * (MAX_CAPACITY + 1) + new_b
                transition_matrix[new_state_index, old_state_index, i] = 1.
    return transition_matrix


# Create transition probability matrix
def create_P_matrix():
    # Request matrix
    P_request_A_one_loc = get_request_transitions_for_one_location(A)
    P_request_A = full_transition_matrix_A(P_request_A_one_loc)
    P_request_B_one_loc = get_request_transitions_for_one_location(B)
    P_request_B = full_transition_matrix_B(P_request_B_one_loc)

    P_request = np.dot(P_request_A, P_request_B)

    # Return matrix
    P_return_A_one_loc = get_return_transition_matrix_one_location(A)
    P_return_A = full_transition_matrix_A(P_return_A_one_loc)
    P_return_B_one_loc = get_return_transition_matrix_one_location(B)
    P_return_B = full_transition_matrix_A(P_return_B_one_loc)

    P_return = np.dot(P_return_A, P_return_B)

    P_return_request = np.dot(P_return, P_request)

    # Nightly move matrix
    P_move = get_nightly_moves()
    P = np.ndarray(shape = ((MAX_CAPACITY + 1) ** 2, (MAX_CAPACITY + 1) ** 2, action_space.shape[0]))

    for i in range(action_space.shape[0]):
        P[:, :, i] = np.dot(P_return_request, P_move[:, :, i])

    return P


# Create reward matrix
def create_R_matrix():
    poisson_mask = np.zeros(shape=(2, MAX_CAPACITY + 1, MAX_CAPACITY + 1))
    po = (poisson.pmf(np.arange(MAX_CAPACITY + 1), REQUEST_RATE[A]),
          poisson.pmf(np.arange(MAX_CAPACITY + 1), REQUEST_RATE[B]))
    for loc in (A, B):
        for i in range(MAX_CAPACITY + 1):
            poisson_mask[loc, i, :i] = po[loc][:i]
            poisson_mask[loc, i, i] = po[loc][i:].sum()
    # The poisson mask contains the probability distribution for renting x cars (x column)
    # in each row j, with j the number of cars available at the location

    reward = np.zeros(shape=(MAX_CAPACITY + 1, MAX_CAPACITY + 1, 2 * TRANSFER_MAX + 1))
    for a in range(MAX_CAPACITY + 1):
        for b in range(MAX_CAPACITY + 1):
            for action in range(-TRANSFER_MAX, TRANSFER_MAX + 1):
                moved_cars = min(action, a) if action >= 0 else max(action, -b)
                a_ = a - moved_cars
                a_ = min(MAX_CAPACITY, max(0, a_))
                b_ = b + moved_cars
                b_ = min(MAX_CAPACITY, max(0, b_))
                reward_a = np.dot(poisson_mask[A, a_], np.arange(MAX_CAPACITY + 1))
                reward_b = np.dot(poisson_mask[B, b_], np.arange(MAX_CAPACITY + 1))
                reward[a, b, action + TRANSFER_MAX] = (
                    (reward_a + reward_b) * RENTAL_INCOME - np.abs(action) * TRANSFER_COST)
    
    reward = reward.reshape(441, 11)
    return reward

# if __name__ == '__main__':
#     print(create_R_matrix().shape)