import numpy as np
from scipy.integrate import solve_ivp

from config import DIR_DATA


class LingchongEqnModel:

    model_name = "lingchong_eqn"

    y0_min = 0.03
    y0_max = 0.07

    alpha_max = 3
    alpha_min = 0.1
    
    k_max = 0.4
    k_min = 0.1

    t_lag_max = 2

    y_max_min = 2.8
    y_max_max = 3.0

    theta_min = 0
    theta_max = 5

    def __init__(
        self,
        n_samples,
        n_timepoints,
        file_prefix,
        t_max=24,
    ):
        self.t_span = (0, t_max,)
        self.t_eval = np.linspace(0, t_max, n_timepoints,)
        self.dir = DIR_DATA / "simulation" / self.model_name
        self.file_prefix = file_prefix
        self.n_samples = n_samples

    def generate_y0(self, n,):
        return np.random.rand(n) * (self.y0_max - self.y0_min) + self.y0_min

    def generate_alpha(self, n,):
        a = np.random.gamma(5, 1, n)
        return ((a.max() - a) / a.max() * self.alpha_max) ** 2 / self.alpha_max + self.alpha_min
    
    def generate_k(self, n,):
        a = np.random.gamma(5, 1, n)
        return ((a.max() - a) / a.max() * self.k_max) ** 2 / self.k_max + self.k_min
    
    def generate_t_lag(self, n,):
        a = np.random.gamma(1, 1, n)
        return a / a.max() * self.t_lag_max
    
    def generate_y_max(self, n,):
        return np.random.rand(n) * (self.y_max_max - self.y_max_min) + self.y_max_min
    
    def generate_theta(self, n,):
        theta = np.random.rand(n) * (self.theta_max - self.theta_min) + self.theta_min
        return theta
    
    def generate_params(self,):
        y0 = self.generate_y0(n=self.n_samples,)
        alpha = self.generate_alpha(n=self.n_samples,)
        k = self.generate_k(n=self.n_samples,)
        t_lag = self.generate_t_lag(n=self.n_samples,)
        y_max = self.generate_y_max(n=self.n_samples,)
        theta = self.generate_theta(n=self.n_samples,)
        np.save(file=self.dir / f"{self.file_prefix}_y0.npy", arr=y0,)
        np.save(file=self.dir / f"{self.file_prefix}_alpha.npy", arr=alpha,)
        np.save(file=self.dir / f"{self.file_prefix}_k.npy", arr=k,)
        np.save(file=self.dir / f"{self.file_prefix}_t_lag.npy", arr=t_lag,)
        np.save(file=self.dir / f"{self.file_prefix}_y_max.npy", arr=y_max,)
        np.save(file=self.dir / f"{self.file_prefix}_theta.npy", arr=theta,)
        return y0, alpha, k, t_lag, y_max, theta
    
    def load_cached_params(self,):
        y0 = np.load(file=self.dir / f"{self.file_prefix}_y0.npy")
        alpha = np.load(file=self.dir / f"{self.file_prefix}_alpha.npy")
        k = np.load(file=self.dir / f"{self.file_prefix}_k.npy")
        t_lag = np.load(file=self.dir / f"{self.file_prefix}_t_lag.npy")
        y_max = np.load(file=self.dir / f"{self.file_prefix}_y_max.npy")
        theta = np.load(file=self.dir / f"{self.file_prefix}_theta.npy")
        return y0, alpha, k, t_lag, y_max, theta

    def fun(self, t, y, alpha, k, y_max, t_lag, theta,):

        dydt = np.zeros_like(y)
        
        if isinstance(t_lag, float) | isinstance(t_lag, int):
            if t >= t_lag:
                return alpha / (1 + (y / k / y_max) ** theta) * (1 - y / y_max) * y
            else:
                return 0
        
        mask = t >= t_lag
        alpha = alpha[mask]
        k = k[mask]
        y_max = y_max[mask]
        y = y[mask]
        theta = theta[mask]

        dydt[mask] = alpha / (1 + (y / k / y_max) ** theta) * (1 - y / y_max) * y
        dydt[~mask] = 0
        return dydt
        
    def solve_ode(self, y0, alpha, k, y_max, t_lag, theta,):
        y0 = y0.flatten()
        
        sol = solve_ivp(
            fun=self.fun,
            t_span=self.t_span,
            y0=y0,
            args=(alpha, k, y_max, t_lag, theta,),
            rtol=1e-6,
            atol=1e-9,
            t_eval=self.t_eval,
        )

        return sol


if __name__ == "__main__":
    USE_CACHE = False

    model = LingchongEqnModel(
        n_samples=10000,
        n_timepoints=128,
        file_prefix="2024-06-06",
    )

    if USE_CACHE:
        y0, alpha, k, t_lag, y_max, theta = model.load_cached_params()
    else:
        y0, alpha, k, t_lag, y_max, theta = model.generate_params()

    sol = model.solve_ode(
        y0=y0,
        alpha=alpha,
        k=k,
        y_max=y_max,
        t_lag=t_lag,
        theta=theta,
    )

    np.savez(
        file=model.dir / f"{model.file_prefix}.npz",
        y_raw=sol.y,
        t=sol.t,
        y=sol.y / sol.y.max(axis=1).reshape(-1, 1),
        y_max=sol.y.max(axis=1).reshape(-1, 1),
        y0=y0,
    )
