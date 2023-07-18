import numpy as np
from race_track_env.race_track import RaceTrack

import matplotlib.pyplot as plt


def behavior_pi(target_pi, epsilon):
    raise NotImplementedError




def off_policy_monte_carlo(track_map:str, render_mode:str):

    gamma = 0.9
    epsilon = 0.1

    env = RaceTrack(track_map, render_mode, size=20)
    action_space = env.nA # (9, ), nine actions in total
    observation_space = env.nS # (curr_row, curr_col, row_speed, col_speed)

    Q = np.zeros(shape=(*observation_space, action_space))
    C = np.zeros_like(Q)
    target_pi = np.argmax(Q, axis=-1)







if __name__ == "__main__":


    off_policy_monte_carlo('a', None)

    # track = 'b'
    # render_mode = None
    # cell_size = 20
    # env = RaceTrack(track, render_mode, cell_size)
    # env.reset()
    # terminated = False
    # total_reward = 0
    # while not terminated:
    #     action = np.random.choice(env.nA)
    #     observation, reward, terminated = env.step(action)
    #     total_reward += reward
    # print(observation, reward, terminated, total_reward)