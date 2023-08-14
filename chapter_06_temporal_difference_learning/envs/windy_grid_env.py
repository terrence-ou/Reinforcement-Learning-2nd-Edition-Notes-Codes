import numpy as np
from gymnasium import Env
import pygame


class WindyGridworld:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, king_move: bool = False, size: int = 2):
        # Initialize gridworld map with cell-wise reward
        self.cols = 10
        self.rows = 7
        # define start and goal states
        self.start = (3, 0)
        self.goal = (3, 7)

        self.map = np.ones(shape=(self.rows, self.cols), dtype=int) * -1
        self.map[self.goal] = 0

        # define wind factors
        self.wind = np.zeros(shape=(self.cols,), dtype=int)
        self.wind[3:9] = 1
        self.wind[6:8] = 2

        # initialize action space
        self.nA = 4 if not king_move else 8
        self.nS = self.map.shape

        self.act_move_map = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, 1),  # RIGHT
            3: (0, -1),  # LEFT
        }

        # Initialize the initial state
        self.reset()

        # Initialize parameters for pygame
        self.window_size = (self.map.shape[1] * size, self.map.shape[0] * size)
        self.window = None  # window for pygame rendering
        self.clock = None  # clock for pygame ticks

    # Take an action and return a tuple of (S', R, Terminated)
    def step(self, action):
        terminated = False
        move = self.act_move_map[action]
        # Adding wind factor to the vertical move
        next_row = self.state[0] + move[0] - self.wind[self.state[1]]
        # check the row's bounds
        next_row = max(0, next_row)
        next_row = min(self.rows - 1, next_row)

        next_col = self.state[1] + move[1]
        # check the column's bounds
        next_col = max(0, next_col)
        next_col = min(self.cols - 1, next_col)

        next_state = (next_row, next_col)

        if next_state == self.goal:
            terminated = True

        self.state = next_state
        reward = self.map[next_state]
        return next_state, reward, terminated

    # Resent environment to the initial state
    def reset(self):
        self.state = (3, 0)
        return self.state

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    test = WindyGridworld()
    test.state = (3, 5)
    print(test.step(2))
    print(test.step(2))
    print(test.step(2))
