import torch

from config import DIR_MODELS
from ml.models.vae_bottleneck import VAEBottleneckModel
from ml.utils.formatting import num_to_str


def upload_vb():
    long_name = "vae-bottleneck"
    short_name = "vb"

    # create model
    print("Creating models")
    for z_dim in [
        2,
        4,
        8,
        16,
        32,
    ]:
        config = {
            "channel2": 128,
            "channel3": 512,
            "channel4": 2048,
            "reduction": int(128 / z_dim),
        }
        model = VAEBottleneckModel(**config)
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
