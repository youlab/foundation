from huggingface_hub import HfApi


def add_to_collection():
    api = HfApi()  # assumes you’ve run `huggingface-cli login`
    collection_slug = "you-lab/microbial-growth-foundation-model-6806eb4da83e9bf22858d913"

    for dataset in [
        "experimental",
        "simulation",
        "processed",
    ]:
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=f"you-lab/foundation-model-data-{dataset}",
            item_type="dataset",
        )
        print(f"Added you-lab/foundation-model-data-{dataset} to {collection_slug}")
