import gym_jcr
import gymnasium

env = gymnasium.make('JacksCarRental-v0')
print(env.reward.shape)