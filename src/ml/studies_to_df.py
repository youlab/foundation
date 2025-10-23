import optuna

from ml.config import (
    OPTUNA_DATABASE,
    OPTUNA_STUDY_NAMES,
)

def main():
    for study_name in OPTUNA_STUDY_NAMES:
        print(f"Working on {study_name}")
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=OPTUNA_DATABASE,
            )
            df = study.trials_dataframe()
            df.to_csv(study_name + ".csv")
        except Exception as e:
            print(f"Exception loading {study_name}: {e}")
