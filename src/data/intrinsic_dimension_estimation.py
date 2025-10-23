
import numpy as np
from sklearn.neighbors import NearestNeighbors


def m_k(
    k,
    d,
):
    return ((k - 1) / np.log(d[:, k].reshape(-1, 1) / d[:, 1:k]).sum(axis=1)).mean()


def calc_distances(
    x,
    k2,
):
    nearest_neighbors = NearestNeighbors(
        n_neighbors=k2+1,
        n_jobs=8,
    ).fit(x)
    distances, _ = nearest_neighbors.kneighbors(x)
    # this mask filters out any duplicates
    return distances[~(distances[:, 1:] == 0).any(axis=1), :]
     

def estimate_intrinsic_dimensions(
    x,
    k1=10,
    k2=20,
):
    distances = calc_distances(
        x=x,
        k2=k2,
    )
    m = x.shape[1]
    while True:
        mk1 = m_k(
            k=k1,
            d=distances,
        )
        if mk1 > m:
            k1 += 10
            k2 += 10
            distances = calc_distances(
                x=x,
                k2=k2,
            )
        else:
            break

    total = 0
    for k in range(k1, k2 + 1,):
        mk = m_k(
            k=k,
            d=distances,
        )
        # print(k, mk)
        if mk > m:
            raise ValueError("this should be avoided based on previous while loop.")
        else:
            total += mk
    total /= k2 - k1 + 1
    return total, k1, k2


def main():
    import pandas as pd
    from tqdm import tqdm

    from config import DIR_RESULTS_DATA
    from data.utils import get_data

    data, idx = get_data()
    y = data["y"]
    k1, k2 = 10, 20

    results = []
    for i, file_name in tqdm(enumerate(idx["experimental"].keys())):
        i1 = idx["experimental"][file_name]["y_all_i"]
        i2 = i1 + idx["experimental"][file_name]["y.shape"][0]
        x = y[i1:i2]
        x = np.unique(x, axis=0)
        if x.shape[0] > k2:
            (
                intrinsic_dimensions,
                final_k1,
                final_k2,
            ) = estimate_intrinsic_dimensions(
                x=x,
                k1=k1,
                k2=k2,
            )
            results.append(
                {
                    "file_name": file_name,
                    "intrinsic_dimensions": intrinsic_dimensions,
                    "k1": final_k1,
                    "k2": final_k2,
                }
            )
        else:
            results.append(
                {
                    "file_name": file_name,
                    "intrinsic_dimensions": -1,
                    "k1": k1,
                    "k2": k2,
                }
            )
    print("Saving first csv")
    pd.DataFrame.from_records(
        data=results,
    ).to_csv(
        DIR_RESULTS_DATA
        / "intrinsic_dimensions.csv",
    )

    k1, k2 = 70, 80
    intrinsic_dimensions, final_k1, final_k2 = estimate_intrinsic_dimensions(
        x=y,
        k1=k1,
        k2=k2,
    )
    results.append(
        {
            "file_name": "all",
            "intrinsic_dimensions": intrinsic_dimensions,
            "k1": final_k1,
            "k2": final_k2,
        }
    )
    print("Saving second csv")
    pd.DataFrame.from_records(
        data=results,
    ).to_csv(
        DIR_RESULTS_DATA
        / "intrinsic_dimensions_all.csv",
    )
