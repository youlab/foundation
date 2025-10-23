import torch

from sklearn.model_selection import train_test_split

from ml.utils.load_models import load_model
from ml.datasets.efficient_datasets import get_efficient_data_loaders
from ml.trainers.mcr_trainer import MCRTrainer
from ml.trainers.pr_trainer import PRTrainer
from ml.trainers.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    model_type,
    z_dim,
    x,
    load_state_dict,
    lr,
    epochs,
):
    model = load_model(
        model_type=model_type,
        z_dim=z_dim,
        load_state_dict=load_state_dict,
    )

    (
        x_train,
        x_test,
    ) = train_test_split(
        x,
        train_size=0.8,
        random_state=42,
    )

    train_loader, test_loader = get_efficient_data_loaders(
        x_train=x_train,
        x_test=x_test,
    )

    if model_type == "MCR":
        trainer = MCRTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
        )

    elif model_type == "PR":
        trainer = PRTrainer(
            model=model,
            train_loader=train_loader,
        )

    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
        )

    trainer.train()

    return trainer
