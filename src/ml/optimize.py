import sys
from datetime import datetime

import optuna

from ml.config import OPTUNA_DATABASE
from ml.train import train
from ml.utils.untrained_models import get_untrained_model


def get_lr(
    trial,
    lr_max=1e-2,
    lr_min=1e-5,
):
    return trial.suggest_float(
        "lr",
        low=lr_min,
        high=lr_max,
        log=True,
    )


def run_training(
    trial,
    model_type,
    z_dim,
    config,
    data_category,
):
    model = get_untrained_model(
        model_type=model_type,
        z_dim=z_dim,
        config=config,
    )
    lr = get_lr(trial=trial)

    return train(
        model_type=model_type,
        z_dim=z_dim,
        model=model,
        lr=lr,
        data_category=data_category,
    )


def objective_vb(
    trial,
):
    data_category = "all"
    model_type = "VB"
    channel2 = 2 ** trial.suggest_int(
        "channel2",
        low=5,
        high=9,
    )
    z_dim = 2 ** trial.suggest_int(
        "z_dim",
        low=1,
        high=5,
        log=True,
    )
    config = {
        "channel2": channel2,
        "channel3": channel2 * 4,
        "channel4": channel2 * 16,
        "reduction": 128 // z_dim,
    }
    return run_training(
        trial=trial,
        model_type=model_type,
        z_dim=z_dim,
        config=config,
        data_category=data_category,
    )
    

def objective_mnm(
    trial,
):
    data_category = "all"
    model_type = "MNM"
    input_dim = 2 ** trial.suggest_int(
        "input_dim",
        low=5,
        high=9,
    )
    hidden_dim = 2 ** trial.suggest_int(
        "hidden_dim",
        low=5,
        high=9,
    )
    reduction = 2 ** trial.suggest_int(
        "reduction",
        low=1,
        high=6,
    )
    num_layers = trial.suggest_int(
        "num_layers",
        low=3,
        high=100,
        log=True,
    )
    z_dim = trial.suggest_int(
        "z_dim",
        low=2,
        high=32,
        log=True,
    )
    config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "z_dim": z_dim,
    }
    return run_training(
        trial=trial,
        model_type=model_type,
        z_dim=z_dim,
        config=config,
        data_category=data_category,
    )


def objective_a7x(
    trial,
):
    data_category = "all"
    model_type = "A7X"
    channel2 = 2 ** trial.suggest_int(
        "channel2",
        low=5,
        high=9,
    )
    reduction = 2 ** trial.suggest_int(
        "reduction",
        low=1,
        high=6,
    )
    z_dim = trial.suggest_int(
        "z_dim",
        low=2,
        high=32,
        log=True,
    )
    config = {
        "channel2": channel2,
        "channel3": channel2 * 4,
        "channel4": channel2 * 16,
        "reduction": reduction,
        "z_dim": z_dim,
    }
    return run_training(
        trial=trial,
        model_type=model_type,
        z_dim=z_dim,
        config=config,
        data_category=data_category,
    )


def get_divisors(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def objective_mcr(
    trial,
):
    data_category = "all"
    model_type = "MCR"
    z_dim = trial.suggest_int(
        "z_dim",
        low=2,
        high=32,
        log=True,
    )
    n_heads = trial.suggest_categorical(
        "n_heads",
        get_divisors(n=z_dim),
    )
    n_layers = 2 ** trial.suggest_int(
        "n_layers",
        low=1,
        high=8,
    )
    
    config = {
        "n_heads": n_heads,
        "n_layers": n_layers,
        "z_dim": z_dim,
    }
    return run_training(
        trial=trial,
        model_type=model_type,
        z_dim=z_dim,
        config=config,
        data_category=data_category,
    )


def main():
    d = datetime.now().date().isoformat()
    objectives = {
        "A7X": objective_a7x,
        "MCR": objective_mcr,
        "MNM": objective_mnm,
        "VB": objective_vb,
    }
    
    model_type = str(sys.argv[1])
    assert model_type in {"A7X", "MCR", "MNM", "VB",}, ValueError(f"Model type incorrect: {model_type}")

    study_name = f"{d}-foundations-{model_type}"
    
    study = optuna.load_study(
        study_name=study_name,
        storage=OPTUNA_DATABASE,
    )

    study.optimize(
        objectives[model_type],
        n_trials=10,
        timeout=260_000,
    )
