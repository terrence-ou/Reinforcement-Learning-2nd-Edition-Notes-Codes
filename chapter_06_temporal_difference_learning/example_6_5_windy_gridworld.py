from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from envs.windy_grid_env import WindyGridworld


# e-greedy policy
def epsilon_greedy(Q: np.ndarray, state: tuple, epsilon: float = 0.1) -> int:
    if np.random.random(size=(1,)) < epsilon:
        return np.random.choice(Q.shape[-1])
    return Q[state].argmax()


def SARSA(env: Any, alpha: float, epsilon: float, total_steps: int = 8000):
    optim_init = 0
    gamma = 1.0

    Q = np.random.normal(loc=optim_init, scale=1.0, size=(*env.nS, env.nA))
    Q[env.goal] = 0  # Set the value of the terminal state to 0

    curr_step = 0
    total_episodes = 0
    step_episode_hist = np.zeros(shape=(total_steps + 1,))
    reward_hist = []

    while curr_step < total_steps:
        episode_reward = 0
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        while True:
            curr_step += 1
            next_state, reward, terminated, truncated = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )
            state = next_state
            action = next_action
            episode_reward += reward
            step_episode_hist[curr_step] = total_episodes
            if terminated or truncated:
                total_episodes += 1
                break
            if curr_step >= total_steps:
                break
    plt.plot(step_episode_hist)
    plt.show()


def run_sarsa_windy():
    raise NotImplementedError


if __name__ == "__main__":
    render_mode = None
    epsilon = 0.1
    alpha = 0.5
    env = WindyGridworld(render_mode=render_mode)
    # env.reset()
    # ttl_reward = 0
    # while True:
    #     next_state, reward, terminated, truncated = env.step(np.random.choice(4))
    #     ttl_reward += reward
    #     if terminated or truncated:
    #         break
    # print(ttl_reward)

    SARSA(env=env, alpha=alpha, epsilon=epsilon)
