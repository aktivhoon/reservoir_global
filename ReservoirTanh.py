from rich.progress import track
import numpy as np

class ReservoirTanh:
    def __init__(self, A, B, C, rs, xs, cs, delT, gam):
        # Matrices
        self.A = A  # N x N matrix of internal reservoir connections
        self.B = B  # N x M matrix of s dynamical inputs to learn
        self.C = C  # N x K matrix of k exeternal inputs for control

        # States and fiexed points
        self.rs = rs
        # check if xs and cs are 1D arrays, and if so, reshape them
        if len(xs.shape) == 1:
            xs = xs.reshape(-1, 1)
        if len(cs.shape) == 1:
            cs = cs.reshape(-1, 1)
        self.xs = xs
        self.cs = cs
        # To ensure non-driven resorvoir begins with stable dynamics,
        # we initialize resovoir parameters by first selecting a distribution of
        # equlibrium points, and then selecting the bias term to achieve that
        # equilibrium point:
        # \dot{r} = 0 = -r + tanh(Ar + Bx + Cc + d)
        self.d = np.arctanh(rs) - np.dot(self.A.A, rs) - np.dot(B, xs) - np.dot(C, cs)

        # Time
        self.delT = delT
        self.gam = gam

        # Initialize reservoir
        self.r = np.zeros((A.A.shape[0], 1))

    def train(self, x, c):
        nx = x.shape[1] - 3
        D = np.zeros((self.A.A.shape[0], nx))
        D[:, 0] = self.r.reshape(-1)
        for i in track(range(1, nx)):
            self.propagate(x[:, i-1:i+3], c)
            D[:, i] = self.r.reshape(-1)
        return D
    
    def propagate(self, x, c):
        # Runge-Kutta 4th order integration
        k1 = self.delT * self.del_r(self.r, x[:, 0], c)
        k2 = self.delT * self.del_r(self.r + 0.5 * k1, x[:, 1], c)
        k3 = self.delT * self.del_r(self.r + 0.5 * k2, x[:, 2], c)
        k4 = self.delT * self.del_r(self.r + k3, x[:, 3], c)
        self.r = self.r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def del_r(self, r, x, c):
        x = x.reshape(-1, 1)
        return self.gam * (-r + np.tanh(np.dot(self.A.A, r) + np.dot(self.B, x) + np.dot(self.C, c) + self.d))

    def predict_x(self, c, W):
        nc = c.shape[1]
        self.r = self.A.A + np.dot(self.B, W)
        D = np.zeros((self.r.shape[0], nc))
        D[:, 0] = self.r.reshape(-1)
        for i in range(1, nc):
            self.propagate_x(c[:, i-1, :])
            D[:, i] = self.r.reshape(-1)
        return D
    
    def propagate_x(self, c):
        # Runge-Kutta 4th order integration
        k1 = self.delT * self.del_r_x(self.r, c[:, 0])
        k2 = self.delT * self.del_r_x(self.r + 0.5 * k1, c[:, 1])
        k3 = self.delT * self.del_r_x(self.r + 0.5 * k2, c[:, 2])
        k4 = self.delT * self.del_r_x(self.r + k3, c[:, 3])
        self.r = self.r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    def del_r_x(self, r, c):
        return self.gam * (-r + np.tanh(np.dot(self.r, r) + np.dot(self.C, c) + self.d))