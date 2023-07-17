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
        
        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        with open('./race_track_env/maps/' + filename, 'rb') as f:
            self.track_map = np.load(f)

        self.window_size = self.track_map.shape
        # Pygame's coordinate if the transpose of that of numpy
        self.window_size = (self.window_size[1] * self.size, self.window_size[0] * self.size)
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
        if self.render_mode == 'human':
            self.render(self.render_mode)
        return self._get_obs(), self._get_info()



    def render(self, mode):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption('Race Track')
            if mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        self.window.fill((255, 255, 255))
        # pygame.draw.rect(self.window, (255, 0, 0), (400, 400, 20, 20), 0)
        
        for row in range(rows):
            for col in range(cols):
                color = (0, 0, 0)
                if self.track_map[row, col] == 0:
                    color = (100, 100, 100)
                elif self.track_map[row, col] == 1:
                    color = (200, 200, 200)
                pygame.draw.rect(self.window, color, (col * self.size, row * self.size, self.size, self.size), 1)

        # if mode == "human":
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass
        self.clock.tick(self.metadata['render_fps'])




if __name__ == '__main__':

    env = RaceTrack('a', render_mode='human', size=20)
    while True:
        env.reset()
    # pygame.init()
    # SCREENWIDTH = 800
    # SCREENHEIGHT = 800
    # RED = (255,0,0)
    # screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

    # pygame.draw.rect(screen, RED, (400, 400, 20, 20),0)
    # screen.fill(RED)

    # pygame.display.update()

    # # waint until user quits
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    # pygame.quit()