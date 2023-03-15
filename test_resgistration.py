import gym 
import os
from gym.envs.registration import register
import numpy as np

register(
    id = "PointmassHard-v0",
    entry_point="envs.pointmass:Pointmass",
    kwargs={"difficulty" : 2}
)
env = gym.make("PointmassHard-v0")
env.set_logdir(os.path.join("logs", "test"))
state, _ = env.reset()
next_state, reward, done, _ = env.step(1)
print(state)

