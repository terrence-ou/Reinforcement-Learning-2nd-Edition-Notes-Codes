import numpy as np
import gymnasium as gym
import pygame


class WindyGridworld:
    def __init__(self, king_move=False):
        # Initialize gridworld map with cell-wise reward
        self.cols = 10
        self.rows = 7
        # define start and goal states
        self.start = (3, 0)
        self.goal = (3, 7)

        self.map = np.ones(shape=(self.rows, self.cols)) * -1
        self.map[self.goal] = 0

        self.wind = np.zeros(shape=(self.cols,))
        self.wind[3:9] = 1
        self.wind[6:8] = 2

        # initialize action space
        self.action_space = 4 if not king_move else 9
        self.act_move_map = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, 1),  # RIGHT
            3: (0, -1),  # LEFT
        }

        # Initialize the initial state
        self.reset()

    # Take an action and return a tuple of (R, S')
    def step(self, action):
        move = self.act_move_map[action]
        next_row = self.state[0] + move[0]
        # Adding wind factor to the vertical move
        next_col = self.state[1] + move[1] - self.wind[self.state[1]]
        if (
            next_row < 0
            or next_row > self.cols - 1
            or next_col < 0
            or next_col > self.cols - 1
        ):
            next_state = self.state
        else:
            next_state = (next_row, next_col)

        self.state = next_state
        reward = self.map[next_state]
        return reward, next_state

    # Resent environment to the initial state
    def reset(self):
        self.state = (3, 0)
        return self.state

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    test = WindyGridworld()
    print(test.map)
    print(test.wind)
