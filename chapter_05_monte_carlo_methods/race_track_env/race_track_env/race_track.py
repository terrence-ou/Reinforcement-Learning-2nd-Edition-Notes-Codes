import os
from os import path
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces


import pygame


class RaceTrack(Env):

    def __init__(self, track_map:str):
        assert track_map in ['a', 'b'], 'track_map value should be "a" or "b"'

        if track_map == 'a':
            pass





if __name__ == '__main__':
    # track_map_path = path.dirname(__file__) + '/maps/track_a.npy'
    print(os.getcwd())
    # with open(track_map_path, 'rb') as f:
        # track = np.load(f)
    
    # print(track.shapes)