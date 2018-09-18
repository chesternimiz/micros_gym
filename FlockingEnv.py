import gym
import gym.spaces as spaces
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class FlockingEnv(gym.Env):
    def __init__(self, num):
        self.size = num
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
        self.state_pos = np.random.uniform(low=-50, high=50, size=(self.size, 2))
        self.state_vel = np.random.uniform(low=-5, high=5, size=(self.size, 2))
        for i in range(0,self.size):
            self.state[i][0] = self.state_pos[i]
            self.state[i][1] = self.state_vel[i]
        return np.array(self.state)

    def update_vel(self,acc):
        for i in range(0,self.size):
            self.state[i][1] += acc*self.delta_t

    def update_pos(self):
        for i in range(0,self.size):
            self.state[i][0] += self.state[i][1]*self.delta_t

    def step(self, action): # action = acc
        self.update_vel(action)
        self.update_pos()
        return np.array(self.state)

    def simple_plot(self):

        # 打开交互模式
        plt.ion()
        self.fig.cla()
        x_index = self.state[:, 0, 0]
        y_index = self.state[:, 0, 1]
        self.fig.scatter(x_index, y_index, marker=".")
        for i in range(0 ,self.size):
            self.fig.annotate("", xy=(x_index[i]+self.state[i, 1, 0], y_index[i]+self.state[i, 1, 1]),
                              xytext=(x_index[i], y_index[i]), arrowprops=dict(arrowstyle="->"))
        plt.pause(0.1)

    def render(self, mode='human'):
        self.simple_plot()


if __name__ == '__main__':

    env = FlockingEnv(50)
    env.reset()
    env.render()
    for i in range(0, 100):
        env.step(np.array([1, 0.5]))
        env.render()
    plt.waitforbuttonpress()

