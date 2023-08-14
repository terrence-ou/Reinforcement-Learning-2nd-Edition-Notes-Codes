from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from envs.windy_grid_env import WindyGridworld


# e-greedy policy
def epsilon_greedy(Q: np.ndarray, state: tuple, epsilon: float = 0.1) -> int:
    if np.random.random(size=(1,)) < epsilon:
        return np.random.choice(Q.shape[-1])
    return Q[state].argmax()


# SARSA Algorithm
def SARSA(env: Any, alpha: float, epsilon: float, total_steps: int = 8000) -> tuple:
    optim_init = 0
    gamma = 1.0

    # Initialize Q values
    # Q = np.random.normal(loc=optim_init, scale=1.0, size=(*env.nS, env.nA))
    Q = np.zeros(shape=(*env.nS, env.nA))
    Q[env.goal] = 0  # Set the value of the terminal state to 0

    curr_step = 0
    total_episodes = 0
    step_episode_hist = np.zeros(shape=(total_steps + 1,))
    reward_hist = []

    # Run SARSA for given steps
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
            # stop conditions
            if terminated or truncated:
                total_episodes += 1
                reward_hist.append(episode_reward)
                break
            if curr_step >= total_steps:
                break

    return Q, step_episode_hist, reward_hist


def run_sarsa_windy(
    epsilon: float, alpha: float, plot: bool = True, render_result: bool = False
) -> None:
    render_mode = None
    env = WindyGridworld(render_mode=render_mode)
    Q, step_episode_hist, reward_hist = SARSA(env, alpha=alpha, epsilon=epsilon)
    if plot:
        plt.plot(step_episode_hist)
        plt.show()
        plt.ylim((-300, 0))
        plt.plot(reward_hist)
        plt.show()

    if render_result:
        A = Q.argmax(axis=-1)
        env = WindyGridworld(render_mode="human")
        state = env.reset()
        while True:
            action = A[state]
            next_state, reward, terminated, truncated = env.step(action)
            state = next_state
            if terminated or truncated:
                break


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.5
    run_sarsa_windy(epsilon=epsilon, alpha=alpha, plot=False, render_result=True)
