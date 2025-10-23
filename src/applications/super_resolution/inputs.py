from applications.super_resolution.down_sample import get_down_sample
from applications.utils.latents import get_latents


def get_inputs(
    y_train,
    y_test,
    mask_pct,
    use_fixed_interval,
    model,
    debug=True,
):

    x_train, x_train_interp = get_down_sample(
        x=y_train,
        mask_pct=mask_pct,
        use_fixed_interval=use_fixed_interval,
    )
    x_test, x_test_interp = get_down_sample(
        x=y_test,
        mask_pct=mask_pct,
        use_fixed_interval=use_fixed_interval,
    )

    x_lat_train = get_latents(
        z=x_train_interp,
        model=model,
    )
    x_lat_test = get_latents(
        z=x_test_interp,
        model=model,
    )
    # y_lat_train removed - we'll predict directly to original space
    if debug:
        print(f"INPUTS: y_train {y_train.shape}, y_test {y_test.shape}, x_train {x_train.shape}, x_test {x_test.shape}, x_lat_train {x_lat_train.shape}, x_lat_test {x_lat_test.shape}")
    return (
        x_train,
        x_test,
        x_test_interp,
        x_lat_train,
        x_lat_test,
        y_train,  # Return y_train in original space instead of y_lat_train
    )
