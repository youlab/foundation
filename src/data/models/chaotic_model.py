"""
Based on the model from this paper: https://www.science.org/doi/10.1126/science.abm7841
"""

import numpy as np
from scipy.integrate import solve_ivp

from config import DIR_DATA


class ChaoticModel:

    model_name = "chaotic"

    alpha_max = 1.
    alpha_min = -1.
    alpha_mean = 0.3
    alpha_std = 0.5

    s = 50

    dispersal = 1e-6
    
    def __init__(
        self,
        file_prefix,
        t_max=500,
    ):
        self.t_span = (0, t_max,)
        self.dir = DIR_DATA / "simulation" / self.model_name
        self.file_prefix = file_prefix

    def generate_params(self,):
        alpha = np.random.normal(
            loc=self.alpha_mean,
            scale=self.alpha_std,
            size=(
                self.s,
                self.s,
            ),
        )
        alpha[alpha > self.alpha_max] = self.alpha_max
        alpha[alpha < self.alpha_min] = self.alpha_min
        np.fill_diagonal(
            alpha,
            1,
        )

        y0 = np.ones(self.s) * 0.1
        np.save(file=self.dir / f"{self.file_prefix}_y0.npy", arr=y0,)
        np.save(file=self.dir / f"{self.file_prefix}_alpha.npy", arr=alpha,)
        return y0, alpha
    
    def load_cached_params(self,):
        y0 = np.load(file=self.dir / f"{self.file_prefix}_y0.npy")
        alpha = np.load(file=self.dir / f"{self.file_prefix}_alpha.npy")
        return y0, alpha

    def fun(self, t, y, alpha,):
        return y * (1 - alpha @ y) + self.dispersal
        
    def solve_ode(self, y0, alpha, t_eval,):
        return solve_ivp(
            fun=self.fun,
            t_span=self.t_span,
            y0=y0,
            args=(alpha,),
            vectorized=True,
            method="Radau",
            t_eval=t_eval,
        )
