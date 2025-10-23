import numpy as np
from scipy.integrate import solve_ivp

from config import DIR_DATA


class AntibioticModel:

    model_name = "antibiotic"

    def __init__(
        self,
        n_samples,
        n_timepoints,
        file_prefix,
        t_max=24,
    ):
        self.d_a = 0.02
        
        self.alpha = None
        self.beta_min = None
        self.xi = None
        self.k_b = None
        self.d_b = None
        self.gamma = None
        self.h_a = 3
        self.inhibitor = None
        self.h_i = 2
        self.phi_max = None
        self.c = None
        self.y0 = None
        self.t_span = (0, t_max,)
        self.t_eval = np.linspace(0, t_max, n_timepoints,)
        self.dir = DIR_DATA / "simulation" / self.model_name
        self.file_prefix = file_prefix
        self.n_samples = n_samples

    def generate_uniform(self, val_min, val_max,):
        return np.random.rand(self.n_samples) * (val_max - val_min) + val_min
    
    def generate_params(self,):
        self.alpha = self.generate_uniform(val_min=0.75, val_max=1.,)
        self.beta_min = self.generate_uniform(val_min=0.1, val_max=1.,)
        self.xi = self.generate_uniform(val_min=0.1, val_max=1.,)
        self.k_b = self.generate_uniform(val_min=0.1, val_max=1.,)
        self.d_b = self.generate_uniform(val_min=1., val_max=10.,)
        self.gamma = self.generate_uniform(val_min=1.1, val_max=1.4,)
        self.inhibitor = self.generate_uniform(val_min=0.1, val_max=10.,)
        self.phi_max = self.generate_uniform(val_min=0.1, val_max=5.,)
        self.c = self.generate_uniform(val_min=0.1, val_max=0.8,)
        self.y0 = np.array([
            self.generate_uniform(val_min=1e-1, val_max=0.4,),
            self.generate_uniform(val_min=1e-1, val_max=0.4,),
            [4] * self.n_samples,
            self.generate_uniform(val_min=1., val_max=100.,),
            [0] * self.n_samples,
        ]).T

        np.save(file=self.dir / f"{self.file_prefix}_alpha.npy", arr=self.alpha)
        np.save(file=self.dir / f"{self.file_prefix}_beta_min.npy", arr=self.beta_min)
        np.save(file=self.dir / f"{self.file_prefix}_xi.npy", arr=self.xi)
        np.save(file=self.dir / f"{self.file_prefix}_k_b.npy", arr=self.k_b)
        np.save(file=self.dir / f"{self.file_prefix}_d_b.npy", arr=self.d_b)
        np.save(file=self.dir / f"{self.file_prefix}_gamma.npy", arr=self.gamma)
        np.save(file=self.dir / f"{self.file_prefix}_inhibitor.npy", arr=self.inhibitor)
        np.save(file=self.dir / f"{self.file_prefix}_phi_max.npy", arr=self.phi_max)
        np.save(file=self.dir / f"{self.file_prefix}_c.npy", arr=self.c)
        np.save(file=self.dir / f"{self.file_prefix}_y0.npy", arr=self.y0)

    def load_cached_params(self,):
        self.alpha = np.load(file=self.dir / f"{self.file_prefix}_alpha.npy",)
        self.beta_min = np.load(file=self.dir / f"{self.file_prefix}_beta_min.npy",)
        self.xi = np.load(file=self.dir / f"{self.file_prefix}_xi.npy",)
        self.k_b = np.load(file=self.dir / f"{self.file_prefix}_k_b.npy",)
        self.d_b = np.load(file=self.dir / f"{self.file_prefix}_d_b.npy",)
        self.gamma = np.load(file=self.dir / f"{self.file_prefix}_gamma.npy",)
        self.inhibitor = np.load(file=self.dir / f"{self.file_prefix}_inhibitor.npy",)
        self.phi_max = np.load(file=self.dir / f"{self.file_prefix}_phi_max.npy",)
        self.c = np.load(file=self.dir / f"{self.file_prefix}_c.npy",)
        self.y0 = np.load(file=self.dir / f"{self.file_prefix}_y0.npy",)

    def fun(self, t, y,):
        y = y.reshape(-1, 5)
        dydt = np.zeros_like(y)

        g = y[:, 2] / (1 + y[:, 2])
        l = self.gamma * y[:, 3] ** self.h_a * g / (1 + y[:, 3] ** self.h_a)
        
        iota = self.inhibitor ** self.h_i / (1 + self.inhibitor ** self.h_i)
        beta = self.beta_min + self.c * (1 - self.beta_min) * iota
        phi = self.phi_max * (1 - self.c * iota)
        
        dydt[:, 0] = (g - l) * y[:, 0]
        dydt[:, 1] = (self.alpha * g - beta * l) * y[:, 1]
        dydt[:, 2] = (self.xi * l - g) * y[:, 0] + (self.xi * beta * l - self.alpha * g) * y[:, 1]
        dydt[:, 3] = (-self.k_b * y[:, 4] - phi * y[:, 1] - self.d_a) * y[:, 3]
        dydt[:, 4] = beta * l * y[:, 1] - self.d_b * iota * y[:, 4]
        
        return dydt.flatten()

    def solve_ode(self):
        sol = solve_ivp(
            fun=self.fun,
            t_span=self.t_span,
            y0=self.y0.flatten(),
            rtol=1e-6,
            atol=1e-9,
            t_eval=self.t_eval,
        )

        return sol


if __name__ == "__main__":
    model = AntibioticModel(
        n_samples=9000,
        n_timepoints=128,
        file_prefix="2024-06-04",
        t_max=16,
    )

    model.load_cached_params()

    sol = model.solve_ode()

    y = sol.y.reshape(model.n_samples, 5, -1)
    
    np.save(
        file=model.dir / f"2025-02-03_y_raw.npy",
        arr=y,
    )

    np.save(
        file=model.dir / f"2025-02-03_t.npy",
        arr=sol.t,
    )
