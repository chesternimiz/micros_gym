import FlockingEnv
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

    def act(self,states):
        self.state = states
        self.calculate_neighbors()

    def get_dist2(self,i,j):
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
                    continue
                n_ij = self.segma_epsilon(q_ij)
                re = re + self.phi_alpha(self.segma_norm(q_ij))*n_ij
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
                


