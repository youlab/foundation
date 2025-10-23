import pickle

from config import DIR_RESULTS_CONSORTIA


def load_data():
    with open(
        DIR_RESULTS_CONSORTIA
        / "consortia_sim_for_figs.pkl",
        "rb",
    ) as fp:
        data = pickle.load(
            file=fp,
        )
    return data
