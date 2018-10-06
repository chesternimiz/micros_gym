import FlockingEnv as fe
import math
import numpy as np


class OSAgent:
    def __init__(self, num, r=25, d=20, h=0.2, a=5, b=5, c1=0.05, c2=0.3, dim=2):
        self.size = num
        self.state = None
        self.neighbors = np.zeros((self.size,self.size),dtype=np.bool)
        self.R = r
        self.D = d
        self.H = h
        self.A = a
        self.B = b
        self.C = abs(self.A-self.B)/math.sqrt(4*self.A*self.B)
        self.C1 = c1
        self.C2 = c2
        self.dim = dim
        self.lp = np.array([0.0, 0.0])
        self.lv = np.array([0.0, 0.0])

    def act(self, states):
        self.state = states
        self.calculate_neighbors()
        # print("f_g:", self.f_g())
        # print("f_d:", self.f_d())
        # print("f_r:", self.f_r())
        return 0.9*self.f_g()+self.f_d()+0.5*self.f_r()

    def get_dist2(self, i, j):
        delta_x = self.state[i][0][0]-self.state[j][0][0]
        delta_y = self.state[i][0][1] - self.state[j][0][1]
        return delta_x*delta_x+delta_y*delta_y

    def calculate_neighbors(self):
        for i in range(0, self.size):
            for j in range(i, self.size):
                if i == j:
                    self.neighbors[i][j] = False
                else:
                    if self.get_dist2(i, j) <= self.R*self.R:
                        self.neighbors[i][j] = True
                        self.neighbors[j][i] = True
                    else:
                        self.neighbors[i][j] = False
                        self.neighbors[j][i] = False

    def f_g(self):
        re = np.zeros((self.size, self.dim), dtype=np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                if j == i:
                    continue
                q_ij = self.state[j][0] - self.state[i][0]
                if np.linalg.norm(q_ij) > self.R:
                # if self.get_dist2(i, j) > self.R*self.R:
                    continue
                n_ij = self.segma_epsilon(q_ij)
                # print("n_ij", n_ij)
                re[i] = re[i] + self.phi_alpha(self.segma_norm(q_ij))*n_ij
        return re

    def segma_epsilon(self, q_ij):
        scale = math.sqrt(1 + 0.1*(q_ij[0]*q_ij[0]+q_ij[1]*q_ij[1]))
        re = q_ij / scale
        return re

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

    def f_d(self):
        re = np.zeros((self.size, self.dim), dtype=np.float32)
        for i in range(0, self.size):
            for j in range(0, self.size):
                if j == i:
                    continue
                q_ij = self.state[j][0] - self.state[i][0]
                if np.linalg.norm(q_ij) > self.R:
                    continue
                p_ij = self.state[j][1] - self.state[i][1]
                re[i] = re[i] + self.a_ij(q_ij)*p_ij
        return re

    def a_ij(self, q_ij):
        r_alpha = self.segma_norm(np.array([self.R, 0]))  # !
        re = self.rho(self.segma_norm(q_ij)/r_alpha)
        return re

    def f_r(self):
        re = np.zeros((self.size, self.dim), dtype=np.float32)
        for i in range(0, self.size):
            re[i] = -self.C1*(self.state[i][0]-self.lp)-self.C2*(self.state[i][1]-self.lv)
        return re

    def set_vl(self, lp, lv):
        self.lp = lp
        self.lv = lv

    def step_vl(self, delta_t):
        self.lp += self.lv * delta_t


if __name__ == '__main__':
    env = fe.FlockingEnv(50, speedup=4)
    agent = OSAgent(50)
    agent.set_vl(lp=np.array([0.0, 0.0]), lv=np.array([1., 1.]))
    state = env.reset()
    env.render()
    for ii in range(0, 1000):
        action = agent.act(states=state)
        # action = np.zeros((2, 2), dtype=np.float32)
        state = env.step(action)
        agent.step_vl(delta_t=env.delta_t)
        if ii % 100 == 0:
            env.render()
    env.wait_button()

