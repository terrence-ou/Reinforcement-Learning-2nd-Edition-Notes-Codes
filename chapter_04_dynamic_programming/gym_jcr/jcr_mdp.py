"""
Still in-progress
"""


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
def get_request_transitions_for_one_location(loc):
    assert (loc == A or loc == B)








if __name__ == '__main__':
    get_request_transitions_for_one_location(0)