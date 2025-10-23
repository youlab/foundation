from applications.datasets.raw_dataset import RawDataset
from applications.datasets.future_simulation_dataset import FutureSimulationDataset

# TODO: get zz_file_paths back in here


def get_data(
    file_path,
    interp_len,
    dir_cache,
    tgt_type,
    model_type,
    model,
    z_dim,
    train_size=None,
    use_raw=False,
    return_raw_data_only=False,
):
    raw_data = RawDataset(
        file_path=file_path,
        interp_len=interp_len,
        train_size=train_size,
    )
    if return_raw_data_only:
        return raw_data
    
    dataset_train = FutureSimulationDataset(
        y=raw_data.y_train,
        n_focal=raw_data.n_focal,
        train_or_test="train",
        train_size=train_size,
        dir_cache=dir_cache,
        use_raw=use_raw,
        tgt_type=tgt_type,
        model=model,
        model_type=model_type,
        z_dim=z_dim,
    )

    dataset_test = FutureSimulationDataset(
        y=raw_data.y_test,
        n_focal=raw_data.n_focal,
        train_or_test="test",
        train_size=train_size,
        dir_cache=dir_cache,
        use_raw=use_raw,
        tgt_type=tgt_type,
        model=model,
        model_type=model_type,
        z_dim=z_dim,
    )

    return raw_data, dataset_train, dataset_test
