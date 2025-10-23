import torch

from config import DIR_MODELS
from ml.models.mlp_network_model import MLPNetworkModel
from ml.utils.formatting import num_to_str


def upload_mnm():
    long_name = "mlp-network-model"
    short_name = "mnm"

    # create model
    print("Creating models")
    for z_dim in [
        2,
        4,
        6,
        8,
        12,
        16,
        20,
        24,
        32,
    ]:
        config = {
            "input_dim": 128,
            "hidden_dim": 512,
            "z_dim": z_dim,
            "num_layers": 3,
        }
        model = MLPNetworkModel(**config)
        model.load_state_dict(
            torch.load(
                DIR_MODELS
                / f"{short_name}_{z_dim}.pth",
            )
        )
        print(f"z={z_dim}, {model}")
        # save locally
        model.save_pretrained(f"{long_name}-{num_to_str(n=z_dim)}")
        print(f"Saved locally")

        # push to the hub
        model.push_to_hub(f"you-lab/{long_name}-{num_to_str(n=z_dim)}")
        print(f"Pushed to the hub")
