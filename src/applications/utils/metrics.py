from config import SEQ_LEN


def get_rmse(
    y_true,
    y_pred,
):
    """
    This function calculates the RMSE for future predictions of consortia.
    
    PARAMETERS
    ----------
    y_true -> np.ndarray : true values
    y_pred -> np.ndarray : predicted values
    out -> tuple : tuple of overall RMSE, RMSE by time point, and RMSE by member by time point
    """
    mse = (y_true[:, :, SEQ_LEN:] - y_pred[:, :, SEQ_LEN:]) ** 2
    return (
        mse.mean() ** 0.5,
        mse.mean(axis=0).mean(axis=0) ** 0.5,
        mse.mean(axis=0) ** 0.5,
    )


def calc_mse(
    y_true,
    y_pred,
):
    return (
        y_true - y_pred
    ) ** 2


def calc_rmse(
    y_true,
    y_pred,
):
    return calc_mse(
        y_true=y_true,
        y_pred=y_pred,
    ).mean() ** 0.5
