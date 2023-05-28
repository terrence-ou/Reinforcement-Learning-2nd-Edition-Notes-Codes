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


if __name__ == '__main__':
    tran = get_request_transitions_for_one_location(0)
    sns.heatmap(tran)
    plt.show()