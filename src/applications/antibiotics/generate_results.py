import logging

from applications.antibiotics.loop_for_results import loop_for_results
from applications.utils.latents import get_latents
from applications.utils.data import cross_val_data_split
from ml.utils.load_models import load_model
from config import SEQ_LEN

logger = logging.getLogger(__name__)


def main(
    x_raw,
    tgt,
    model_type,
    z_dim,
    classify,
    detailed_binary,
    epochs_fine_tuned,
    lr_fine_tuned,
    epochs_end2end,
    lr_end2end,
    train_sizes=None,
    return_latent_vectors=False,
    stack=1,
    cross_val=-1,
    k_fold=5,
    run_fine_tuned_and_end2end=False,
    prediction_model_cache_dir=None,
    name_suffix=None,
    max_depth=None,
):
    if not isinstance(stack, int):
        raise TypeError(f"stack should be int but is {type(stack)}")
    if stack < 1:
        raise ValueError(f"stack should be >= 1 but is {stack}")

    if not isinstance(cross_val, int):
        raise TypeError(f"cross_val should be int but is {type(cross_val)}")
    if (cross_val < 0) or (cross_val > 4):
        raise ValueError(f"cross_val should be >= 0 and <= 4 but is {cross_val}")

    logger.info("Loading model and getting latents")
    model = load_model(
        model_type=model_type,
        z_dim=z_dim,
    )

    x_latents = get_latents(
        z=x_raw.reshape(-1, SEQ_LEN),
        use_transformer=model_type == "MCR",
        z_dim=z_dim,
        model=model,
    ).reshape(x_raw.shape[0], -1,)

    (
        x_raw_train_original,
        _,
    ) = cross_val_data_split(
        x=x_raw,
        n_cross_val=cross_val,
        k_fold=k_fold,
    )
    
    if stack != 1:
        x_raw_train_original = x_raw_train_original.reshape(-1, SEQ_LEN)

    x_latents_fine_tuned = None
    x_latents_end2end = None

    logger.info("Splitting data and getting results")
    if return_latent_vectors:
        return (
            split_data_and_get_results(
                x_raw=x_raw,
                x_latents=x_latents,
                x_latents_fine_tuned=x_latents_fine_tuned,
                x_latents_end2end=x_latents_end2end,
                tgt=tgt,
                classify=classify,
                detailed_binary=detailed_binary,
                cross_val=cross_val,
                prediction_model_cache_dir=prediction_model_cache_dir,
                name_suffix=name_suffix,
                max_depth=max_depth,
                train_sizes=train_sizes,
            ),
            x_latents,
            x_latents_fine_tuned,
            x_latents_end2end,
        )
    return split_data_and_get_results(
        x_raw=x_raw,
        x_latents=x_latents,
        x_latents_fine_tuned=x_latents_fine_tuned,
        x_latents_end2end=x_latents_end2end,
        tgt=tgt,
        classify=classify,
        detailed_binary=detailed_binary,
        cross_val=cross_val,
        prediction_model_cache_dir=prediction_model_cache_dir,
        name_suffix=name_suffix,
        train_sizes=train_sizes,
        max_depth=max_depth,
    )


def split_data_and_get_results(
    x_raw,
    x_latents,
    x_latents_fine_tuned,
    x_latents_end2end,
    tgt,
    classify,
    detailed_binary,
    train_sizes=None,
    cross_val=-1,
    k_fold=5,
    prediction_model_cache_dir=None,
    name_suffix=None,
    max_depth=None,
):
    (
        x_raw_train_original,
        x_raw_test,
    ) = cross_val_data_split(
        x=x_raw,
        n_cross_val=cross_val,
        k_fold=k_fold,
    )

    (
        x_latent_train_original,
        x_latent_test,
    ) = cross_val_data_split(
        x=x_latents,
        n_cross_val=cross_val,
        k_fold=k_fold,
    )

    x_latent_fine_tuned_train_original = None
    x_latent_fine_tuned_test = None
    x_latent_end2end_train_original = None
    x_latent_end2end_test = None

    (
        tgt_train_original,
        tgt_test,
    ) = cross_val_data_split(
        x=tgt,
        n_cross_val=cross_val,
        k_fold=k_fold,
    )

    logger.info("Looping for results")
    return loop_for_results(
        x_raw_train_original=x_raw_train_original,
        x_raw_test=x_raw_test,
        x_latent_train_original=x_latent_train_original,
        x_latent_test=x_latent_test,
        x_latent_fine_tuned_train_original=x_latent_fine_tuned_train_original,
        x_latent_fine_tuned_test=x_latent_fine_tuned_test,
        x_latent_end2end_train_original=x_latent_end2end_train_original,
        x_latent_end2end_test=x_latent_end2end_test,
        tgt_train_original=tgt_train_original,
        tgt_test=tgt_test,
        classify=classify,
        detailed_binary=detailed_binary,
        prediction_model_cache_dir=prediction_model_cache_dir,
        name_suffix=name_suffix,
        cross_val=cross_val,
        train_sizes=train_sizes,
        max_depth=max_depth,
    )
