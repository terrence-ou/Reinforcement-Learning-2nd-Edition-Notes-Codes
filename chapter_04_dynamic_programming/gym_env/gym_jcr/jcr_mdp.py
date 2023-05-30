'''
IMPORTANT:
The whole MDP part of the Jack's Car Rental environment is based on Gertjan gsverhoeven's 
implementation with MIT Liscence:
https://github.com/gsverhoeven/gym_jcr/blob/main/gym_jcr/jcr_mdp.py
'''


import numpy as np
from scipy.stats import poisson

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
        transition_matrix[i * block_size : (i * block_size) + block_size,
                          i * block_size : (i * block_size) + block_size] = transition_one_loc
    
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
    P_return_B = full_transition_matrix_B(P_return_B_one_loc)

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




# Function tests
if __name__ == '__main__':
    
    state_ = np.zeros([MAX_CAPACITY+1, MAX_CAPACITY+1])
    state_[11, 15] = 1. # 11 cars at A and 15 cars at B
    state_11_15 = state_.reshape(-1)
    print('\nTesting functions: ')


    #  ===== Function test: get_request_transitions_for_one_location =====
    P_request_A_one_loc = get_request_transitions_for_one_location(A)
    # all colums should sum to one
    np.testing.assert_allclose(P_request_A_one_loc.sum(axis=0), 1.)
    print('[PASSED]: get_request_transitions_for_one_location')


    #  ===== Function test: full_transition_matrix_A =====
    P_request_A = full_transition_matrix_A(P_request_A_one_loc)
    np.testing.assert_almost_equal(np.dot(P_request_A, state_11_15).reshape(MAX_CAPACITY+1,-1).sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_request_A, state_11_15).reshape(MAX_CAPACITY+1,-1)[:,15].sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_request_A, state_11_15).reshape(MAX_CAPACITY+1,-1)[:12,15].sum(), 1.)
    print('[PASSED]: full_transition_matrix_A')


    #  ===== Function test: full_transition_matrix_B =====
    P_request_B_one_loc = get_request_transitions_for_one_location(1)
    P_request_B = full_transition_matrix_B(P_request_B_one_loc)
    np.testing.assert_almost_equal(np.dot(P_request_B, state_11_15).reshape(MAX_CAPACITY+1,-1).sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_request_B, state_11_15).reshape(MAX_CAPACITY+1,-1)[11].sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_request_B, state_11_15).reshape(MAX_CAPACITY+1,-1)[11,:16].sum(), 1.)
    print('[PASSED]: full_transition_matrix_B')


    #  ===== Function test: get_return_transitions_for_one_location =====
    P_return_A_one_loc = get_return_transition_matrix_one_location(A)
    np.testing.assert_almost_equal(P_return_A_one_loc.sum(axis=0), 1.)
    P_return_A = full_transition_matrix_A(P_return_A_one_loc)

    # should mix only states of A: 
    np.testing.assert_almost_equal(np.dot(P_return_A, state_11_15).reshape(MAX_CAPACITY+1,-1).sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_return_A, state_11_15).reshape(MAX_CAPACITY+1,-1)[:,15].sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_return_A, state_11_15).reshape(MAX_CAPACITY+1,-1)[11:,15].sum(), 1.) 
    
    P_return_B_one_loc = get_return_transition_matrix_one_location(B)
    P_return_B = full_transition_matrix_B(P_return_B_one_loc)
    # should mix only states of B: 
    np.testing.assert_almost_equal(np.dot(P_return_B, state_11_15).reshape(MAX_CAPACITY+1,-1).sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_return_B, state_11_15).reshape(MAX_CAPACITY+1,-1)[11].sum(), 1.)
    np.testing.assert_almost_equal(np.dot(P_return_B, state_11_15).reshape(MAX_CAPACITY+1,-1)[11,15:].sum(), 1.)
    print('[PASSED]: get_return_transitions_for_one_location')



    #  ===== Function test: get_nightly_moves =====
    P_move = get_nightly_moves()
    np.testing.assert_allclose(P_move.sum(axis=0), 1.)
    # check some moves
    # assert P_move[:,21*car_at_A+cars_at_B,action+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[new_cars_at_A, new_cars_at_B] == 1.
    assert P_move[:,0,0].reshape(MAX_CAPACITY+1, -1)[0,0] == 1.

    # e.g. from state [1,0] and action 1 => new state should be  [0,1]
    assert P_move[:,21*1+0,1+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[0,1] == 1. 
    assert P_move[:,21*1+1,-2+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[2,0] == 1. 
    assert P_move[:,21*9+5,0+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[9,5] == 1. 
    assert P_move[:,21*9+5,3+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[6,8] == 1. 
    assert P_move[:,21*9+5,-3+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[12,2] == 1.
    assert P_move[:,21*20+20,5+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[15,20] == 1. 
    assert P_move[:,21*20+20,-4+TRANSFER_MAX].reshape(MAX_CAPACITY+1, -1)[20,16] == 1. 
    print('[PASSED]: get_nightly_moves\toutputshape: {}'.format(P_move.shape))


    #  ===== Function test: get_nightly_moves =====
    P = create_P_matrix()
    np.testing.assert_almost_equal(P.sum(axis=0), 1.)
    print('[PASSED]: create_P_matrix')

