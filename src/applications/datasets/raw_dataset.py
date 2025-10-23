import numpy as np
from scipy.interpolate import interp1d


class RawDataset:
    n_samples = 10_000

    def __init__(
        self,
        file_path,
        interp_len,
        train_size,
    ):
        self.y = np.loadtxt(file_path)
        if str(file_path).find("100K") > -1:
            self.n_samples = 100_000
        elif str(file_path).find("bgLV/I1/bgLV_B96_T4_random") > -1:
            raise ValueError("Sample size too small")
            self.n_samples = 1_000
        elif str(file_path).find("bgLV/I7/bgLV_B16_T4_random") > -1:
            raise ValueError("Sample size too small")
            self.n_samples = 1_000
        elif str(file_path).find("bgLV/I4/bgLV_B46_T4_random") > -1:
            raise ValueError("Sample size too small")
            self.n_samples = 1_000

        elif str(file_path).find("dgLV/I4/dgLV_B95_T5_fixed") > -1:
            self.n_samples = 50_000
        elif str(file_path).find("dgLV/I4/dgLV_B98_T2_fixed") > -1:
            self.n_samples = 50_000
        elif str(file_path).find("dgLV/I4/dgLV_B97_T3_fixed") > -1:
            self.n_samples = 50_000
        elif str(file_path).find("dgLV/I4/dgLV_B92_T8_fixed") > -1:
            self.n_samples = 50_000
        elif str(file_path).find("dgLV/I3/dgLV_B95_T5_fixed") > -1:
            self.n_samples = 20_000
        elif str(file_path).find("dgLV/I3/dgLV_B97_T3_fixed") > -1:
            self.n_samples = 20_000
        elif str(file_path).find("dgLV/I3/dgLV_B92_T8_fixed") > -1:
            self.n_samples = 20_000
        elif str(file_path).find("dgLV/I3/dgLV_B98_T2_fixed") > -1:
            self.n_samples = 20_000

        self.n_focal = int(str(file_path).split("/")[-1].split("_T")[1].split("_")[0])
        if not (self.n_focal == self.y.shape[0] // self.n_samples):
            print(
                f"Error with {file_path}, focal populations should be {self.n_focal} but array size is {self.y.shape}"
            )
            raise ValueError(
                f"Error with {file_path}, focal populations should be {self.n_focal} but array size is {self.y.shape}"
            )

        x2 = np.arange(interp_len).astype(float)
        x1 = (
            np.linspace(
                0,
                interp_len - 1,
                self.y.shape[1],
            )
            .astype(int)
            .astype(float)
        )

        f = interp1d(
            x1,
            self.y,
        )

        self.y = f(x2)
        self.y[self.y > 1] = 1
        self.y[self.y < 0] = 0

        train_size = train_size * self.n_focal
        self.y_train = self.y[:train_size, :]
        test_size = 2_000 * self.n_focal
        self.y_test = self.y[-test_size:, :]
