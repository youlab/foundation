import torch

from config import DIR_MODELS
from ml.models.autoencoder7x import Autoencoder7XModel
from ml.utils.formatting import num_to_str


def upload_a7x():
    long_name = "autoencoder7x"
    short_name = "a7x"

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
            "input_length": 128,
            "channel2": 128,
            "channel3": 512,
            "channel4": 2048,
            "reduction": 16,
            "z_dim": z_dim,
        }
        model = Autoencoder7XModel(**config)
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
