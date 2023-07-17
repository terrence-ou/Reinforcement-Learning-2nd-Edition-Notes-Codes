import os
from os import path
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces

import pygame


class RaceTrack(Env):

    def __init__(self, track_map:str):
        assert track_map in ['a', 'b'], 'track_map value should be "a" or "b"'

        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        with open('./race_track_env/maps/' + filename, 'rb') as f:
            self.track_map = np.load(f)

        # Get start states
        self.start_states = np.dstack(np.where(self.track_map==0.8))[0]

        self.P = {}
        self.nA = 9
        # self.nS = 



if __name__ == '__main__':

    env = RaceTrack('a')