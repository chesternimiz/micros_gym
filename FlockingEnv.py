import math
import gym
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np


class FlockingEnv(gym.Env):
    def __init__(self, num):
        self.size = num
        self.action_space = \
            spaces.Box(low=-1000, high=1000, shape=[num, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=[num, 2, 2], dtype=np.float32)
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)


if __name__ == '__main__':
    env = FlockingEnv(50)
    print('ok')
