import numpy as np
from scipy.integrate import solve_ivp


class GrowthModel:
    def __init__(
        self,
        n,
        mu_min=0.1,
        mu_max=2.0,
        y0_const=0.1,
        t_end=8,
        t_points=100,
    ):
        self.n = n
        self.mu = np.linspace(
            mu_min,
            mu_max,
            self.n,
        )
        self.y0 = np.zeros_like(self.mu) + y0_const
        self.t_span = (0, t_end)
        self.t_eval = np.linspace(0, self.t_span[1], t_points)

    def fun(
        self,
        t,
        y,
    ):
        return self.mu * y * (1 - y)

    def solve_ode(self):
        return solve_ivp(
            fun=self.fun,
            t_span=self.t_span,
            y0=self.y0,
            t_eval=self.t_eval,
        )
