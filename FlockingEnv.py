import gym
import gym.spaces as spaces
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


class FlockingEnv(gym.Env):
    def __init__(self, num, r=25, speedup=1, col_dist=0.1):
        self.size = num
        self.R = r
        self.col_dist = col_dist
        self.speedup = speedup
        self.action_space = \
            spaces.Box(low=-1000, high=1000, shape=[num, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=[num, 2, 2], dtype=np.float32)
        self.state_pos = None
        self.state_vel = None
        self.state = np.zeros((num, 2, 2), dtype=np.float32)
        self.delta_t = 0.1
        plt.ion()
        self.fig = plt.figure().add_subplot(1, 1, 1)
        self.fig.axis("equal")

    def reset(self):
        self.state_pos = np.random.uniform(low=-100, high=100, size=(self.size, 2))
        self.state_vel = np.random.uniform(low=-1, high=1, size=(self.size, 2))
        for i in range(0, self.size):
            self.state[i][0] = self.state_pos[i]
            self.state[i][1] = self.state_vel[i]
        return np.array(self.state)

    def reset_mul(self):
        self.reset()
        observation = np.zeros((self.size,self.size, 2, 2),dtype=np.float32)
        for i in range(0,self.size):
            for j in range(0,self.size):
                q_ij = observation[j][0] - observation[i][0]
                if np.linalg.norm(q_ij) <= self.R:
                    observation[i][j] = self.state[j]
        return observation

    def update_vel(self, acc):
        self.state[:, 1] += acc*self.delta_t
        #for i in range(0, self.size):
        #    self.state[i][1] += acc*self.delta_t

    def update_pos(self):
        for i in range(0, self.size):
            self.state[i][0] += self.state[i][1]*self.delta_t

    def step(self, action):  # action = acc
        self.update_vel(action)
        self.update_pos()
        return np.array(self.state)

    def step_mul(self, action):
        self.step(action)
        observation = np.zeros((self.size, self.size, 2, 2),dtype=np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                q_ij = observation[j][0] - observation[i][0]
                if np.linalg.norm(q_ij) <= self.R:
                    observation[i][j] = self.state[j]
        reward = np.zeros(self.size,dtype=np.float32)
        return observation

    def simple_plot(self):

        # 打开交互模式
        plt.ion()
        self.fig.cla()
        x_index = self.state[:, 0, 0]
        y_index = self.state[:, 0, 1]
        self.fig.scatter(x_index, y_index, marker=".")
        for i in range(0, self.size):
            self.fig.annotate("", xy=(x_index[i]+self.state[i, 1, 0]*3, y_index[i]+self.state[i, 1, 1]*3),
                              xytext=(x_index[i], y_index[i]), arrowprops=dict(arrowstyle="->",color="blue"))
            for j in range(i+1, self.size):
                if np.linalg.norm(self.state[i][0]-self.state[j][0]) <= self.R:
                    self.fig.annotate("", xy=(x_index[i], y_index[i]),
                                      xytext=(x_index[j], y_index[j]), arrowprops=dict(arrowstyle="-"))

        plt.pause(0.1/self.speedup)

    def render(self, mode='human'):
        self.simple_plot()

    def wait_button(self):
        plt.waitforbuttonpress()


if __name__ == '__main__':

    env = FlockingEnv(50)
    action = np.zeros((50, 2), dtype=np.float32)
    for i in range(0, 50):
        action[i][0] = 1
        action[i][1] = 0.5
    env.reset()
    env.render()
    for i in range(0, 100):
        env.step(np.array(action))
        env.render()
    env.wait_button()

