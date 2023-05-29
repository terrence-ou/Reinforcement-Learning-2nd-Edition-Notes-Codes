import gym_jcr
import gymnasium
import numpy as np

MAX_CAPACITY = 20
TRANSFER_MAX = 5

def get_transition_kernel_for_policy(policy):
    return P[:, range((MAX_CAPACITY + 1) ** 2), policy + TRANSFER_MAX]



if __name__ == "__main__":
    env = gymnasium.make('JacksCarRental-v0')
    R = env.reward
    P = env.transition
    print('Reward shape:', R.shape)
    print('Transition prob. shape:', P.shape)

    
    policy = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.int32)
    values = np.zeros(shape=((MAX_CAPACITY+1) ** 2), dtype=np.float32)
    # P = P.transpose(2, 0, 1)


    P_pi = get_transition_kernel_for_policy(policy)