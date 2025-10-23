import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from config import PATH_FUJITA_DATA
from data.normalization_functions.utils import get_t_cols


class RawDataset:
    def __init__(
        self,
        test_reps,
        source_file,
        strains,
        interp_mult: int=6,
    ):
        
        self.source_file = source_file
        self.strains = strains
        self.n_focal = len(self.strains)
        interp_len = 128 * interp_mult
        df = pd.read_csv(PATH_FUJITA_DATA)
        t_cols, _ = get_t_cols(df)
        for file in df.file.unique():
            for media in df.media.unique():
                for rep in df.rep.unique():
                    mask = (
                        df.file == file
                    ) & (
                        df.media == media
                    ) & (
                        df.rep == rep
                    )
                    if mask.sum() == 0:
                        continue
                    x = df.loc[mask, t_cols]
                    x /= x.sum(axis=0)
                    df.loc[mask, t_cols] = x

        df = df.loc[df.file == source_file].copy()
        df = df.loc[[strain in self.strains for strain in df.strain]].copy()
        y = df.loc[:, t_cols].to_numpy()
        y = y / y.max(axis=1).reshape(-1, 1)

        x2 = np.arange(interp_len).astype(float)
        x1 = (
            np.linspace(
                0,
                interp_len - 1,
                y.shape[1],
            )
            .astype(int)
            .astype(float)
        )

        f = interp1d(
            x1,
            y,
        )

        self.y = f(x2)
        self.key = df.loc[:, ["file", "media", "rep", "strain",]].copy()

        self.y_train, self.y_test = self.get_samples(test_reps=test_reps)

    def get_samples(
        self,
        test_reps,
    ):
        train = []
        for i in range(self.y.shape[0]):
            if self.key.rep.iloc[i] in test_reps:
                train.append(False)
            else:
                train.append(True)
        return (
            self.y[np.array(train), :],
            self.y[~np.array(train), :],
        )
