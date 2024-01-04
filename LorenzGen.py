import numpy as np
import jax.numpy as jnp

class Lorenz:
    def __init__(self, x0, delT, params=[10, 28, 2.667]):
        self.x = x0
        self.delT = delT
        self.s, self.r, self.b = params

        # Check the type of x0 to determine whether to use NumPy or JAX
        if isinstance(x0, np.ndarray):
            self.np = np
        else:
            self.np = jnp

    def propagate(self, n):
        xs = self.np.empty((n+1, self.x.shape[0]))
        # print self.x type
        if self.np == np:
            xs[0] = self.x
        else:
            xs = xs.at[0].set(self.x)
        for i in range(n):
            x_dot = self.s * (self.x[1] - self.x[0])
            y_dot = self.x[0] * (self.r - self.x[2]) - self.x[1]
            z_dot = self.x[0] * self.x[1] - self.b * self.x[2]
            self.x += self.np.array([x_dot, y_dot, z_dot]) * self.delT
            if self.np == np:
                xs[i+1] = self.x
            else:
                xs = xs.at[i+1, :].set(self.x)
        # xs has a shape of (dimensions, time_steps + 1)
        xs = xs.T
        return xs