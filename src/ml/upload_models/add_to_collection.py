from huggingface_hub import HfApi


def num_to_str(
    n,
):
    if n < 10:
        return f"0{n}"
    return str(n)


def add_to_collection():
    api = HfApi()  # assumes you’ve run `huggingface-cli login`
    collection_slug = "you-lab/microbial-growth-foundation-model-6806eb4da83e9bf22858d913"

    for model_type in [
        # "autoencoder7x",
        # "mlp-network-model",
        # "microbert-curve-reducer",
        # "vae-bottleneck",
        "pca-reducer",
    ]:
        if model_type == "vae-bottleneck":
            z_dims = [2, 4, 8, 16, 32,]
        elif model_type == "microbert-curve-reducer":
            z_dims = [6, 8, 12, 16, 20, 24, 32,]
        else:
            z_dims = [2, 4, 6, 8, 12, 16, 20, 24, 32,]

        for z_dim in z_dims:
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=f"you-lab/{model_type}-{num_to_str(n=z_dim)}",
                item_type="model",
            )
            print(f"Added you-lab/{model_type}-{num_to_str(n=z_dim)} to {collection_slug}")
