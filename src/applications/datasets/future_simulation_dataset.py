import numpy as np

from applications.utils.latents import get_latents
from config import (
    SEQ_LEN,
    Z_DIM,
)


class FutureSimulationDataset:
    def __init__(
        self,
        y,
        n_focal,
        train_or_test,
        train_size,
        tgt_type,
        model,
        model_type,
        z_dim,
        save_cache=False,
        use_cache=False,
        dir_cache=None,
        use_raw=False,
        step_size=32,
    ):
        self.n_focal = n_focal
        self.use_x_raw = use_raw
        self.step_size = step_size
        self.tgt_type = tgt_type

        file_name = (
            f"future_plus_one_data_{train_or_test}_{train_size}_{step_size}_{tgt_type}.npz"
        )

        try:
            if not use_cache:
                raise ValueError
            data = np.load(dir_cache / file_name)
            self.tgt_ = data["tgt"]
            self.x_latents = data["x_latents"]
            self.x_raw = data["x_raw"]

        except:
            y[y < 0] = 0
            self.x_raw = []
            self.tgt_raw = []

            if self.tgt_type.find("segment") > -1:
                window_shift = int(self.tgt_type[7:])
                for i in range(0, y.shape[1] - SEQ_LEN - window_shift, self.step_size):
                    self.x_raw.append(y[:, i:i + SEQ_LEN])
                    self.tgt_raw.append(y[:, i + SEQ_LEN:i + SEQ_LEN + window_shift])
                self.tgt_raw = np.array(self.tgt_raw).reshape(-1, window_shift,)
                self.tgt_raw = self.tgt_raw.reshape(
                    -1,
                    self.n_focal,
                    window_shift,
                )
                self.tgt_latents = np.array([0])
            elif self.tgt_type.find("sliders") > -1:
                window_shift = int(self.tgt_type[7:])
                for i in range(0, y.shape[1] - SEQ_LEN - window_shift, self.step_size):
                    self.x_raw.append(y[:, i:i + SEQ_LEN])
                    self.tgt_raw.append(y[:, i + window_shift:i + SEQ_LEN + window_shift])
                self.tgt_raw = np.array(self.tgt_raw).reshape(-1, SEQ_LEN,)
                self.tgt_latents = get_latents(
                    z=self.tgt_raw,
                    use_transformer=model_type == "MCR",
                    z_dim=z_dim,
                    model=model,
                )
                self.tgt_raw = self.tgt_raw.reshape(
                    -1,
                    self.n_focal,
                    SEQ_LEN,
                )
                self.tgt_latents = self.tgt_latents.reshape(
                    -1,
                    self.n_focal,
                    Z_DIM + 1,
                )
            elif self.tgt_type == "latent":
                for i in range(0, y.shape[1] - SEQ_LEN * 2, self.step_size):
                    self.x_raw.append(y[:, i:i + SEQ_LEN])
                    self.tgt_raw.append(y[:, i + SEQ_LEN:i + SEQ_LEN * 2])
                self.tgt_raw = np.array(self.tgt_raw).reshape(-1, SEQ_LEN,)
                self.tgt_latents = get_latents(
                    z=self.tgt_raw,
                    use_transformer=model_type == "MCR",
                    z_dim=z_dim,
                    model=model,
                )
                self.tgt_raw = self.tgt_raw.reshape(
                    -1,
                    self.n_focal,
                    SEQ_LEN,
                )
                self.tgt_latents = self.tgt_latents.reshape(
                    -1,
                    self.n_focal,
                    Z_DIM + 1,
                )

            else:
                self.tgt_latents = np.array([0])
                if self.tgt_type == "nextval":
                    for i in range(0, y.shape[1] - SEQ_LEN, self.step_size):
                        self.x_raw.append(y[:, i:i + SEQ_LEN])
                        self.tgt_raw.append(y[:, i + SEQ_LEN])
                elif self.tgt_type == "log":
                    y[y<1e-9] = 1e-9
                    for i in range(0, y.shape[1] - SEQ_LEN, self.step_size):
                        self.x_raw.append(y[:, i:i + SEQ_LEN])
                        self.tgt_raw.append(np.log10(y[:, i + SEQ_LEN]))
                elif self.tgt_type == "dydt":
                    max_val = 100
                    for i in range(0, y.shape[1] - SEQ_LEN, self.step_size):
                        self.x_raw.append(y[:, i:i + SEQ_LEN])
                        y0 = y[:, i + SEQ_LEN - 1:i + SEQ_LEN]
                        y1 = y[:, i + SEQ_LEN:i + SEQ_LEN + 1]
                        dydt = (y1 - y0) / y0 * 100
                        dydt[(y0 == 0) & (y1 == 0)] = 0
                        dydt[(y0 == 0) & (y1 != 0)] = max_val
                        dydt[dydt > max_val] = max_val
                        dydt[dydt < -max_val] = -max_val
                        self.tgt_raw.append(dydt)
                self.tgt_raw = np.array(self.tgt_raw).reshape(-1, 1)
                self.tgt_raw = self.tgt_raw.reshape(
                    -1,
                    self.n_focal,
                    1,
                )
                
            self.x_raw = np.array(self.x_raw).reshape(-1, SEQ_LEN)
            if self.use_x_raw:
                self.x_latents = np.zeros(1)
            else:
                self.x_latents = get_latents(
                    z=self.x_raw[:, :SEQ_LEN],
                    use_transformer=model_type == "MCR",
                    z_dim=z_dim,
                    model=model,
                )

                self.x_latents = np.concatenate((self.x_latents, self.x_raw[:, SEQ_LEN:]), axis=1,)
                self.x_latents = self.x_latents.reshape(
                    -1,
                    self.n_focal,
                    Z_DIM + 1,
                )
                self.x_latents[np.isnan(self.x_latents)] = 0

            self.x_raw = self.x_raw.reshape(
                -1,
                self.n_focal,
                SEQ_LEN,
            )

            if save_cache:
                np.savez(
                    dir_cache / file_name,
                    tgt_raw=self.tgt_raw,
                    tgt_latents=self.tgt_latents,
                    x_raw=self.x_raw,
                    x_latents=self.x_latents,
                )

    @property
    def tgt(self):
        if (self.tgt_type == "latent") or (self.tgt_type.find("sliders") > -1):
            return self.tgt_latents
        return self.tgt_raw

    @property
    def x(self):
        if self.use_x_raw:
            return self.x_raw
        return self.x_latents
