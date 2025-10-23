import numpy as np
import pandas as pd
from scipy.io import loadmat

from applications.antibiotics.carolyn.config import (
    datasets,
    fowler_labels_1,
    fowler_labels_2,
    anderson_labels,
)
from config import DIR_DATA_CAROLYN


def load_data_and_apply_labels(
    files,
    labels,
):
    data = []
    for fn in files:
        data.append(loadmat(DIR_DATA_CAROLYN / "blanked_data" / fn)["datFin"])
    data = np.concatenate(data)
    data[data < 0] = 0
    data = pd.DataFrame(data)
    data["strain"] = labels
    return data


def get_data():
    df = pd.concat(
        (
            load_data_and_apply_labels(
                files=datasets["10000"][:10],
                labels=fowler_labels_1.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["10000"][10:],
                labels=anderson_labels.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["lambda_MOI1_10000"][:10],
                labels=fowler_labels_2.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["lambda_MOI1_10000"][10:],
                labels=anderson_labels.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["100"][:10],
                labels=fowler_labels_2.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["100"][10:],
                labels=anderson_labels.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["Carb-5ugml_10000"][:10],
                labels=fowler_labels_2.repeat(4),
            ),
            load_data_and_apply_labels(
                files=datasets["Carb-5ugml_10000"][10:],
                labels=anderson_labels.repeat(4),
            ),
        ),
    )

    resistance_label = loadmat(DIR_DATA_CAROLYN / "data_dependency_files" / "resistancesLabel.mat")["resistancesFowlerAnderson"]
    resistances = {}
    for i, strain in enumerate(fowler_labels_2):
        resistances[strain] = resistance_label[i]

    for i, strain in enumerate(anderson_labels):
        resistances[strain] = resistance_label[i+218]

    df["SAM"] = np.nan
    df["GM"] = np.nan
    df["SXT"] = np.nan
    df["CIP"] = np.nan
    antibiotics = ["SAM", "GM", "SXT", "CIP",]
    for strain in df.strain.unique():
        mask = df.strain == strain
        for i, antibiotic in enumerate(antibiotics):
            df.loc[mask, antibiotic] = resistances[strain][i]
    
    return df
