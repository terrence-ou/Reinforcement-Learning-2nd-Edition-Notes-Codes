import os
from os import path
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces

import pygame


class RaceTrack(Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, track_map:str, render_mode:str=None, size:int=2):
        self.size = size
        self.window_size = 512
        
        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_mode']
        self.render_mode = render_mode

        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        with open('./race_track_env/maps/' + filename, 'rb') as f:
            self.track_map = np.load(f)

        # Get start states
        self.start_states = np.dstack(np.where(self.track_map==0.8))[0]

        # Define the observation space and action space
        # self.observation_space = {
        #     'row': spaces.Discrete(self.track_map.shape[0]),
        #     'col': spaces.Discrete(self.track_map.shape[1]),
        #     'speed_row': spaces.Discrete(9), # range(-4, 5)
        #     'speed_col': spaces.Discrete(9), # range(-4, 5)
        # }
        # self.action_space = spaces.Discrete(9)

        # Define the shape of observations and actions
        self.nS = (*self.track_map.shape, 9, 9)
        self.nA = 9
        self.state = None # Initialize state
        self.speed = None # Initialize speed

        self._action_to_acceleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

        self.window = None
        self.clock = None


    def _get_obs(self):
        return (*self.state, *self.speed)

    def _get_info(self):
        return None
    
    
    def reset(self):
        start_idx = np.random.choice(self.start_states.shape[0])
        self.state = self.start_states[start_idx]
        self.speed = (0, 0)
        return self._get_obs(), self._get_info()



if __name__ == '__main__':

    env = RaceTrack('a')
    print(env.reset())