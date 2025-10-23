import torch

from config import DIR_MODELS
from ml.config import CONFIG_MCR
from ml.models.microbert_curve_reducer import MicroBERTCurveReducerModel
from ml.utils.formatting import num_to_str


def upload_mcr():
    long_name = "microbert-curve-reducer"
    short_name = "mcr"

    # create model
    print("Creating models")
    for z_dim in [
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        6,
        12,
        20,
        24,
    ]:
        config = {
            "dimension": z_dim,
            "n_heads": CONFIG_MCR[z_dim]["n_heads"],
            "n_layers": CONFIG_MCR[z_dim]["n_layers"],
        }
        model = MicroBERTCurveReducerModel(**config)
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
