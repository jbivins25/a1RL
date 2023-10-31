import gym
import gym_a1
import numpy as np
import time

env = gym.make('a1-v0')
'''
#Call env.step(action) in a loop here to take actions
#Call env.reset() when you feel it's reasonable to reset the gym environment (for example if the robot falls over).
obs = env.reset()
while True:
    action = env.action_space.sample()
    
    obs, reward, done, info = env.step(action)
    
    if done:
        break
env.close()

print("Ran Successfully")'''
