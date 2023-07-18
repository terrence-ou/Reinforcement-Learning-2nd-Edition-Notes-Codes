import numpy as np
from race_track_env.race_track import RaceTrack

import matplotlib.pyplot as plt


def behavior_pi(state:tuple, 
                nA:int, 
                target_pi:np.ndarray, 
                epsilon:float) -> tuple:
    '''
    The behaviro policy returns both the action and 
    the probability of that action
    '''

    rand_val = np.random.rand()
    greedy_act = target_pi[state]
    
    if rand_val > epsilon:
        return greedy_act, (1 - epsilon + epsilon / nA)
    else:
        action = np.random.choice(nA)
        if action == greedy_act:
            return action, (1 - epsilon + epsilon / nA)
        else:
            return action, epsilon / nA



def off_policy_monte_carlo(track_map:str, render_mode:str):

    gamma = 0.9
    epsilon = 0.1
    total_episodes = 100_000

    env = RaceTrack(track_map, render_mode, size=20)
    action_space = env.nA # (9, ), nine actions in total
    observation_space = env.nS # (curr_row, curr_col, row_speed, col_speed)

    Q = np.random.normal(size=(*observation_space, action_space)) - 500
    C = np.zeros_like(Q)
    target_pi = np.argmax(Q, axis=-1)

    for i in range(total_episodes):
        
        # Generate a trajectory using behaviro policy
        trajectory = []
        terminated = False
        state, info = env.reset()
        (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)
        
        ttl_reward = 0

        while not terminated:
            observation, reward, terminated = env.step(action)
            ttl_reward += reward
            trajectory.append((state, action, reward, act_prob))
            state = observation
            (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)
        
        G = 0
        W = 1.

        while trajectory:
            (state, action, reward, act_prob) = trajectory.pop()
            G = gamma * G + reward
            C[state][action] = C[state][action] + W
            Q[state][action] = Q[state][action] + (W / C[state][action]) * (G - Q[state][action])

            target_pi[state] = np.argmax(Q[state])
            if action != target_pi[state]:
                break
            W = W * (1 / act_prob)
        
        if i % 1000 == 0:
            print(f'Episode: {i}, reward: {ttl_reward}, epsilon: {epsilon}')


if __name__ == "__main__":


    off_policy_monte_carlo('a', None)