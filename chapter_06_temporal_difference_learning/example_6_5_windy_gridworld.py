import numpy as np
from envs.windy_grid_env import WindyGridworld
from typing import Any


# e-greedy policy
def epsilon_greedy(Q: np.ndarray, state: tuple, epsilon: float = 0.1) -> int:
    if np.random.random(size=(1,)) < epsilon:
        return np.random.choice(Q.shape[-1])
    return Q[state].argmax()


def SARSA(env: Any, alpha: float, epsilon: float):
    raise NotImplementedError


def run_sarsa_windy():
    raise NotImplementedError


if __name__ == "__main__":
    render_mode = "human"
    env = WindyGridworld(render_mode=render_mode)
    env.reset()
    while True:
        next_state, reward, terminated, truncated = env.step(np.random.choice(4))
        if terminated or truncated:
            break
