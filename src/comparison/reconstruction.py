import pandas as pd
from sklearn.metrics import r2_score

from applications.utils.latents import reconstruct
from compare_models.config import NOW_TEXT
from config import DIR_RESULTS_MODEL_COMPARISON
from data.utils import get_data
from ml.utils.load_models import load_model


def get_recons(
    train_all,
    test_all,
    train_sim,
    test_sim,
    train_exp,
    test_exp,
    model,
    batch_size,
    return_recons=False,
    is_vae=True,
    is_pca=False,
):
    recon_train_all = reconstruct(
        model=model,
        x=train_all,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )
    recon_test_all = reconstruct(
        model=model,
        x=test_all,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )

    recon_train_sim = reconstruct(
        model=model,
        x=train_sim,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )
    recon_test_sim = reconstruct(
        model=model,
        x=test_sim,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )

    recon_train_exp = reconstruct(
        model=model,
        x=train_exp,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )
    recon_test_exp = reconstruct(
        model=model,
        x=test_exp,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )

    if return_recons:
        return (
            recon_train_all,
            recon_test_all,
        )

    r2_train_all = r2_score(
        y_true=train_all.flatten(),
        y_pred=recon_train_all.flatten(),
    )
    
    r2_test_all = r2_score(
        y_true=test_all.flatten(),
        y_pred=recon_test_all.flatten(),
    )

    r2_train_sim = r2_score(
        y_true=train_sim.flatten(),
        y_pred=recon_train_sim.flatten(),
    )
    
    r2_test_sim = r2_score(
        y_true=test_sim.flatten(),
        y_pred=recon_test_sim.flatten(),
    )

    r2_train_exp = r2_score(
        y_true=train_exp.flatten(),
        y_pred=recon_train_exp.flatten(),
    )
    
    r2_test_exp = r2_score(
        y_true=test_exp.flatten(),
        y_pred=recon_test_exp.flatten(),
    )

    return (
        r2_train_all,
        r2_test_all,
        r2_train_sim,
        r2_test_sim,
        r2_train_exp,
        r2_test_exp,
    )


def main(
    model_type,
    z_dim,
    batch_size=4_096,
    check_cache=False,
):
    if check_cache:
        try:
            pd.read_csv(
                DIR_RESULTS_MODEL_COMPARISON
                / f"model_recon_{model_type}_{z_dim}_{NOW_TEXT}.csv",
            )
            return
        except:
            pass
    (
        train_all,
        test_all,
    ) = get_data(
        category="all",
        return_split=True,
    )
    (
        train_exp,
        test_exp,
    ) = get_data(
        category="experimental",
        return_split=True,
    )
    (
        train_sim,
        test_sim,
    ) = get_data(
        category="simulation",
        return_split=True,
    )

    model = load_model(
        model_type=model_type,
        z_dim=z_dim,
    )

    (
        r2_train_all,
        r2_test_all,
        r2_train_sim,
        r2_test_sim,
        r2_train_exp,
        r2_test_exp,
    ) = get_recons(
        train_all=train_all,
        test_all=test_all,
        train_sim=train_sim,
        test_sim=test_sim,
        train_exp=train_exp,
        test_exp=test_exp,
        model=model,
        batch_size=batch_size,
        is_vae=model_type.find("vae") > -1,
        is_pca=model_type.find("PR") > -1,  # FIXME: these need to be addressed
    )
    pd.DataFrame.from_records(
        [
            {
                "model_name": f"{model_type}_{z_dim}",
                "r2_train_all": r2_train_all,
                "r2_test_all": r2_test_all,
                "r2_train_sim": r2_train_sim,
                "r2_test_sim": r2_test_sim,
                "r2_train_exp": r2_train_exp,
                "r2_test_exp": r2_test_exp,
            },
        ]
    ).to_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / f"model_recon_{model_type}_{z_dim}_{NOW_TEXT}.csv",
    )
