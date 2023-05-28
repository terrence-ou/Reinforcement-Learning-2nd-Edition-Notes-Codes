import numpy as np
import sys
from six import StringIO, b

import gymnasium as gym
from gymnasium import utils, spaces

from scipy.stats import poisson
from .jcr_mdp import *

MAX_CARS = 20
MAX_MOVE_OF_CARS = 5

Ptrans = create_P_matrix()
R = create_R_matrix()



class JacksCarRentalEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, render_mode=None):
        print("")

        actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)
        nA = len(actions)
        nS = (MAX_CARS + 1) ** 2
        P = {
            s: {
                a: [
                    (Ptrans[next_s, s, a], next_s, R[s, a], False)
                    for next_s in range(nS)
                ]
                for a in range(nA)
            }
            for s in range(nS)
        }
        # isd - initial state dist
        isd = np.full(nS, 1 / nS)
        observation_shape = (MAX_CARS + 1, MAX_CARS + 1)

        self.observation_shape = observation_shape
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        self.nS = nS
        self.nA = nA
        self.P = P
        self.isd = isd
        self.transition = Ptrans.transpose(1, 2, 0)
        self.reward = R
