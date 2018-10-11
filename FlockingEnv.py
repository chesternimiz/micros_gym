import gym
import gym.spaces as spaces
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

potential_force=[]
potential = []

class OSReward:
    def __init__(self, num, r=25, d=20, col_dist=0.1,  h=0.2, a=5, b=5):
        self.size=num
        self.R = r
        self.D = d
        self.col = col_dist
        self.H = h
        self.A = a
        self.B = b
        self.C = abs(self.A - self.B) / math.sqrt(4 * self.A * self.B)
        self.d_alpha = self.segma_norm(np.array([self.D, 0]))
        self.r_alpha = self.segma_norm(np.array([self.R, 0]))
        self.potential_step=0.1


    def calculate_potential(self):
        start = 0
        end = int(self.r_alpha/self.potential_step)
        while start <= end:
            pf = self.phi_alpha(start*self.potential_step)
            # print (start,pf)
            potential_force.append(pf)
            potential.append(0.0)
            start += 1
        print("potential force calculation finished")
        begin = int(self.d_alpha/self.potential_step)
        start = begin
        p_value = 0.0
        while start <= end:
            p_value += potential_force[start]*self.potential_step
            potential[start] = p_value
            start += 1
        print("potential forward finished with", end, potential[end])
        start = begin
        p_value = 0
        while start >= 0:
            p_value -= potential_force[start] * self.potential_step
            potential[start] = p_value
            start -= 1
        print("potential backward finished with", 0, potential[0])

    def potential_reward(self,q_ij):
        re = 0
        z = self.segma_norm(q_ij)
        '''
        if z < self.d_alpha:
            step = -0.1
            start = self.d_alpha
            end = z
            while start > end:
              re += self.phi_alpha(start)*step
              start+=step
        else:
            step = 0.1
            start = self.d_alpha
            end = z
            while start < end and start<self.r_alpha:
                re += self.phi_alpha(start) * step
                start+=step
        '''
        if z < self.r_alpha:
            re = potential[int(z/self.potential_step)]
        else:
            re = potential[int(self.r_alpha/self.potential_step)]
        return -re

    def consensus_reward(self,p_ij):
        return -0.5*np.linalg.norm(p_ij)

    def vl_reward(self,p,q):
        return -0.5*np.linalg.norm(q)-0.5*np.linalg.norm(p)

    def segma_norm(self, q_ij):
        re = (math.sqrt(0.1*(q_ij[0]*q_ij[0]+q_ij[1]*q_ij[1])+1)-1)/0.1
        return re

    def phi_alpha(self,seg):
        r_alpha = self.segma_norm(np.array([self.R, 0]))  # !
        d_alpha = self.segma_norm(np.array([self.D, 0]))  # !
        re = self.rho(seg/r_alpha)*self.phi(seg-d_alpha)
        return re

    def rho(self, z):
        if z < self.H:
            return 1.0
        else:
            if z > 1:
                return 0
            else:
                return 0.5*(1+math.cos(math.pi*(z-self.H)/(1-self.H)))

    def phi(self, z):
        return 0.5*((self.A+self.B)*self.segma_1(z+self.C)+self.A-self.B)

    def segma_1(self, z):
        return z / math.sqrt(1+z*z)

class FlockingEnv(gym.Env):
    def __init__(self, num, r=25, speedup=1, col_dist=0.1, dynamic="second"):
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
        self.osr = OSReward(self.size)
        self.vlq=np.array([100.,100.])
        self.vlp=np.array([1.,1.])
        self.osr.calculate_potential()
        self.dynamic = dynamic
        self.last_reward = None

    def reset(self):
        self.state_pos = np.random.uniform(low=0, high=200, size=(self.size, 2))
        self.state_vel = np.random.uniform(low=-1, high=1, size=(self.size, 2))
        for i in range(0, self.size):
            self.state[i][0] = self.state_pos[i]
            self.state[i][1] = self.state_vel[i]
        return np.array(self.state)

    def reset_mul(self):
        self.reset()
        observation = np.zeros((self.size,self.size+1, 2, 2),dtype=np.float32)
        for i in range(0,self.size):
            for j in range(0,self.size):
                q_ij = observation[j][0] - observation[i][0]
                if np.linalg.norm(q_ij) <= self.R:
                    observation[i][j] = self.state[j]
            observation[i][self.size][0] = self.vlq
            observation[i][self.size][1] = self.vlp
        self.last_reward = self.get_reward()
        return observation

    def update_vel(self, acc):
        if self.dynamic == "second":
            self.state[:, 1] += acc*self.delta_t
        else:
            self.state[:,1] = acc
        #for i in range(0, self.size):
        #    self.state[i][1] += acc*self.delta_t

    def update_pos(self):
        for i in range(0, self.size):
            self.state[i][0] += self.state[i][1]*self.delta_t

    def step(self, action1):  # action = acc
        self.update_vel(action1)
        self.update_pos()
        '''
        p_r=np.zeros(self.size,np.float32)
        c_r=np.zeros(self.size,np.float32)
        v_r=np.zeros(self.size,np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                q_ij = self.state[j][0] - self.state[i][0]
                if np.linalg.norm(q_ij) <= self.R:
                    p_r[i]+=self.osr.potential_reward(q_ij)
                    p_ij=self.state[j][1] - self.state[i][1]
                    c_r[i]+=self.osr.consensus_reward(p_ij)
            q=self.state[i][0]-self.vlq
            p=self.state[i][0]-self.vlp
            v_r[i]=self.osr.vl_reward(p,q)
        print(p_r.sum(),c_r.sum(),v_r.sum(),p_r.sum()+c_r.sum()+v_r.sum())
        '''
        return np.array(self.state)

    def step_mul(self, action2):
        self.step(action2)
        self.vlq += self.vlp*self.delta_t
        observation = np.zeros((self.size, self.size+1, 2, 2),dtype=np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                q_ij = observation[j][0] - observation[i][0]
                if np.linalg.norm(q_ij) <= self.R:
                    observation[i][j] = self.state[j]
            observation[i][self.size][0] = self.vlq
            observation[i][self.size][1] = self.vlp
        '''
        p_r = np.zeros(self.size, np.float32)
        c_r = np.zeros(self.size, np.float32)
        v_r = np.zeros(self.size, np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                q_ij = self.state[j][0] - self.state[i][0]
                p_r[i] += self.osr.potential_reward(q_ij)
                if np.linalg.norm(q_ij) <= self.R:
                    p_ij = self.state[j][1] - self.state[i][1]
                    c_r[i] += self.osr.consensus_reward(p_ij)
            q = self.state[i][0] - self.vlq
            p = self.state[i][0] - self.vlp
            v_r[i] = self.osr.vl_reward(p, q)
        '''
        new_reward = self.get_reward()
        reward = new_reward - self.last_reward
        self.last_reward = new_reward
        # print(p_r.sum(), c_r.sum(), v_r.sum(), reward.sum())
        info=new_reward.sum()
        return observation,reward,False,info

    def get_reward(self):
        p_r = np.zeros(self.size, np.float32)
        c_r = np.zeros(self.size, np.float32)
        v_r = np.zeros(self.size, np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                q_ij = self.state[j][0] - self.state[i][0]
                p_r[i] += self.osr.potential_reward(q_ij)
                if np.linalg.norm(q_ij) <= self.R:
                    p_ij = self.state[j][1] - self.state[i][1]
                    c_r[i] += self.osr.consensus_reward(p_ij)
            q = self.state[i][0] - self.vlq
            p = self.state[i][0] - self.vlp
            v_r[i] = self.osr.vl_reward(p, q)
        reward = p_r + c_r + v_r
        return reward

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

