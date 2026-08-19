CONFIG_MCR = {
    2: {
        "n_heads": 2,
        "n_layers": 12,
        "lr": 0.001,
    },
    4: {
        "n_heads": 2,
        "n_layers": 8,
        "lr": 0.0001,
    },
    6: {
        "n_heads": 2,
        "n_layers": 8,
        "lr": 0.0001,
    },
    8: {
        "n_heads": 4,
        "n_layers": 12,
        "lr": 0.0001,
    },
    12: {
        "n_heads": 12,
        "n_layers": 12,
        "lr": 0.0001,
    },
    16: {
        "n_heads": 4,
        "n_layers": 16,
        "lr": 0.0001,
    },
    20: {
        "n_heads": 10,
        "n_layers": 8,
        "lr": 0.001,
    },
    24: {
        "n_heads": 2,
        "n_layers": 16,
        "lr": 0.0001,
    },
    32: {
        "n_heads": 2,
        "n_layers": 16,
        "lr": 0.0001,
    },
}

CONFIG_VB = {
    "channel2": 128,
    "channel3": 512,
    "channel4": 2048,
    "lr": 3e-5,
}

CONFIG_A7X = {
    "input_length": 128,
    "channel2": 128,
    "channel3": 512,
    "channel4": 2048,
    "reduction": 16,
    "lr": 1e-4,
}

CONFIG_MNM = {
    "input_dim": 128,
    "hidden_dim": 512,
    "num_layers": 3,
    "lr": 4e-5,
}

# empirically, training usually converges well within 1000 epochs
TRAINING_EPOCHS = 1000

OPTUNA_DATABASE = "sqlite:///core2.db"
OPTUNA_STUDY_NAMES = []

# this determines the directory containing the processed corpus (post train/test split)
# used when training models, i.e. running run_train_ml.sh, which runs run_ml.py with RUN_TRAIN=True
DATA_RUN_DIR = "perc_sim_100"

# this determines the directory containing the trained model
# used when evaluating models, i.e. running run_vae_posttrain_eval.sh, which runs run_vae_posttrain_eval.py
MODEL_RUN_DIR = "a7x_08_run_2026-05-05T01:11:54.987230_all_1000epochs"
