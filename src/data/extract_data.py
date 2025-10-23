import os
from pathlib import Path

import pandas as pd

DIR_DATA = Path.home() / "foundations" / "src" / "data" / "experimental" / "chory_lab"

for root, directories, files in os.walk(DIR_DATA):
    if root == '/hpc/home/zah8/foundations/src/data/experimental/chory_lab':
        continue
    for fn in files:
        if fn.find("Absorbances.csv") > -1:
            fn_new = root.split("/")[-2] + "_" + root.split("/")[-1] + "_" + fn.split(".csv")[0]
            df = pd.read_csv(f"{root}/{fn}")            
            df = df.drop_duplicates()
            if len(df.Time.unique()) < 10:
                continue
            for plate_id in df.plate_id.unique():
                df.loc[df.plate_id == plate_id, :].pivot(
                    columns="Time",
                    index="well",
                    values="reading",
                ).to_csv(
                DIR_DATA / f"{fn_new}_{plate_id}.csv",
            )
