import jax
import jax.numpy as jnp
from jax import jit
from rich.progress import track

@jit
def _propagate(r, x, c, delT, gam, A, B, C, d):
    k1 = delT * _del_r(r, x[:, 0], c, gam, A, B, C, d)
    k2 = delT * _del_r(r + 0.5 * k1, x[:, 1], c, gam, A, B, C, d)
    k3 = delT * _del_r(r + 0.5 * k2, x[:, 2], c, gam, A, B, C, d)
    k4 = delT * _del_r(r + k3, x[:, 3], c, gam, A, B, C, d)
    r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r

@jit
def _del_r(r, x, c, gam, A, B, C, d):
    x = x.reshape(-1, 1)
    return gam * (-r + jnp.tanh(jnp.dot(A, r) + jnp.dot(B, x) + jnp.dot(C, c) + d))

class JaxReservoirTanh:
    def __init__(self, A, B, C, rs, xs, cs, delT, gam):
        self.A = A
        self.B = B
        self.C = C
        self.rs = rs

        if len(xs.shape) == 1:
            xs = xs.reshape(-1, 1)
        if len(cs.shape) == 1:
            cs = cs.reshape(-1, 1)

        self.xs = xs
        self.cs = cs
        self.d = jnp.arctanh(rs) - jnp.dot(A, rs) - jnp.dot(B, xs) - jnp.dot(C, cs)
        self.delT = delT
        self.gam = gam
        self.r = jnp.zeros((A.shape[0], 1))

    def train(self, x, c):
        nx = x.shape[1] - 3
        D = jnp.zeros((self.A.shape[0], nx))
        D = D.at[:, 0].set(self.r.reshape(-1))
        for i in track(range(1, nx)):
            self.propagate(x[:, i-1:i+3], c)
            D = D.at[:, i].set(self.r.reshape(-1))
        return D

    def propagate(self, x, c):
        self.r = _propagate(self.r, x, c, self.delT, self.gam, self.A, self.B, self.C, self.d).block_until_ready()
        return self.r

    def del_r(self, r, x, c):
        return _del_r(r, x, c, self.gam, self.A, self.B, self.C, self.d)

    def predict_x(self, c, W):
        nc = c.shape[1]
        self.r = self.A + jnp.dot(self.B, W)
        D = jnp.zeros((self.r.shape[0], nc))
        D = jax.ops.index_update(D, jax.ops.index[:, 0], self.r.reshape(-1))
        for i in range(1, nc):
            self.propagate_x(c[:, i-1, :])
            D = jax.ops.index_update(D, jax.ops.index[:, i], self.r.reshape(-1))
        return D

    @jit
    def propagate_x(self, c):
        k1 = self.delT * self.del_r_x(self.r, c[:, 0])
        k2 = self.delT * self.del_r_x(self.r + 0.5 * k1, c[:, 1])
        k3 = self.delT * self.del_r_x(self.r + 0.5 * k2, c[:, 2])
        k4 = self.delT * self.del_r_x(self.r + k3, c[:, 3])
        self.r = self.r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @jit
    def del_r_x(self, r, c):
        return self.gam * (-r + jnp.tanh(jnp.dot(self.r, r) + jnp.dot(self.C, c) + self.d))