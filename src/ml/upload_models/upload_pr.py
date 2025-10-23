from huggingface_hub import Repository

from config import DIR_MODELS
from ml.utils.formatting import num_to_str


def upload_pr():
    long_name = "pca-reducer"
    short_name = "pr"

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
        # clone-or-init
        repo = Repository(
            local_dir=f"{long_name}-{num_to_str(n=z_dim)}",
            clone_from=f"you-lab/{long_name}-{num_to_str(n=z_dim)}",
            use_auth_token=True,
        )

        # add your model file and a README
        repo.git_add(DIR_MODELS / f"{short_name}_{z_dim}.joblib")

        # commit & push
        repo.push_to_hub("Upload sklearn model")
