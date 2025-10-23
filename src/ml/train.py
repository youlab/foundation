import joblib
import os
from datetime import datetime

import numpy as np
import torch

from config import DIR_RESULTS_MODEL
from data.utils import get_data
from ml.config import (
    CONFIG_A7X,
    CONFIG_MCR,
    CONFIG_MNM,
    CONFIG_VB,
    TRAINING_EPOCHS,
)
from ml.datasets.efficient_datasets import get_efficient_data_loaders
from ml.trainers.mcr_trainer import MCRTrainer
from ml.trainers.trainer import Trainer
from ml.utils.formatting import num_to_str
from ml.utils.untrained_models import get_untrained_model


def run_trainer(
    model_name,
    trainer=None,
    model=None,
):
    dir_model = DIR_RESULTS_MODEL / model_name
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    if trainer is not None:
        test_loss = trainer.train()
        print("Saving model", test_loss)

        torch.save(
            obj=trainer.model.state_dict(),
            f=dir_model / f"model.pth",
        )

        np.save(
            file=dir_model / f"train_loss.npy",
            arr=np.array(trainer.train_loss),
        )

        np.save(
            file=dir_model / f"test_loss.npy",
            arr=np.array(trainer.test_loss),
        )
        return test_loss

    elif model is not None:
        
        print("Saving model")
        joblib.dump(
            value=model,
            filename=dir_model / f"model.joblib",
        )


def train(
    model_type,
    z_dim,
    model,
    lr,
    data_category,
):
    print("Loading data")
    (
        x_train,
        x_test,
    ) = get_data(
        category=data_category,
        return_split=True,
    )

    (
        train_loader,
        test_loader,
    ) = get_efficient_data_loaders(
        x_train=x_train,
        x_test=x_test,
    )

    if model_type in {"A7X", "MNM", "VB",}:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=TRAINING_EPOCHS,
            lr=lr,
        )
        test_loss = run_trainer(
            model_name=f"{model_type.lower()}_{num_to_str(n=z_dim)}_{datetime.now().isoformat()}_{data_category}",
            trainer=trainer,
        )
    elif model_type == "MCR":
        trainer = MCRTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=TRAINING_EPOCHS,
            lr=lr,
        )
        test_loss = run_trainer(
            model_name=f"{model_type.lower()}_{num_to_str(n=z_dim)}_{datetime.now().isoformat()}_{data_category}",
            trainer=trainer,
        )
    elif model_type == "PR":
        for x in train_loader:
            model.partial_fit(x.cpu().detach().numpy()[:, 0, :])
        test_loss = run_trainer(
            model_name=f"{model_type.lower()}_{num_to_str(n=z_dim)}_{datetime.now().isoformat()}_{data_category}",
            model=model,
        )

    return test_loss


def main(
    model_type,
    z_dim,
    config=None,
    data_category="all",
):
    if config is None:
        if model_type == "A7X":
            config = CONFIG_A7X
            config["z_dim"] = z_dim
        elif model_type == "MCR":
            config = CONFIG_MCR[z_dim]
            config["dimension"] = z_dim
        elif model_type == "MNM":
            config = CONFIG_MNM
            config["z_dim"] = z_dim
        elif model_type == "VB":
            config = CONFIG_VB
            config["reduction"] = 128 // z_dim
    elif config.get("lr") is not None:
        raise ValueError(f"Config must contain learning rate")

    model = get_untrained_model(
        model_type=model_type,
        z_dim=z_dim,
        config=config,
    )

    train(
        model_type=model_type,
        z_dim=z_dim,
        model=model,
        lr=config["lr"],
        data_category=data_category,
    )
