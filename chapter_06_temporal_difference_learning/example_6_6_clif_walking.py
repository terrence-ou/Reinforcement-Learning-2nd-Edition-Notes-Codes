import numpy as np
import gymnasium as gym


def SARSA(env):
    raise NotImplementedError


def Q_learning(env, epsilon, alpha, gamma):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))
    state, info = env.reset()


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.5
    gamma = 1.0

    env = gym.make("CliffWalking-v0")
    Q_learning(env, epsilon, alpha, gamma)
