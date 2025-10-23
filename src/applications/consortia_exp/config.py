from itertools import product


def get_params(task_id=None):
    interp_lens = [
        480,
        3200,
    ]
    n_memberses = [
        2,
        3,
        5,
        8,
    ]
    train_sizes = [
        10,
        40,
        100,
    ]

    max_curves = [
        2_000_000,
        10_000_000,
        50_000_000,
    ]

    tgt_types = [
        # "nextval",
        # "log",
        # "dydt",
        "segment16",
        "segment32",
        "segment128",
    ]
    use_raws = [
        False,
        True,
    ]
    
    params = list(
        product(
            train_sizes,
            n_memberses,
            interp_lens,
            max_curves,
            tgt_types,
            use_raws,
        )
    )

    if task_id is None:
        return params
    return params[task_id]


def get_config(source_file):
    strain_def = {
        "soil-a": [
            "X_0003",
            "X_0007",
            "X_0008",
            "X_0013",
            "X_0014",
            "X_0016",
            "X_0022",
            "X_0027",
            "X_0036",
            "X_0043",
            "X_0053",
            "X_0055",
            "X_0058",
            "X_0063",
            "X_0073",
            "X_0079",
            "X_0105",
            "X_0195",
        ],
        "soil-b": [
            "X_0001",
            "X_0005",
            "X_0006",
            "X_0008",
            "X_0015",
            "X_0018",
            "X_0031",
            "X_0037",
            "X_0043",
            "X_0086",
            "X_0101",
        ],
        "soil-c": [
            "X_0004",
            "X_0007",
            "X_0008",
            "X_0015",
            "X_0027",
            "X_0053",
            "X_0097",
            "X_0112",
        ],
        "water-a": [
            "X_0003",
            "X_0013",
            "X_0016",
            "X_0020",
            "X_0022",
            "X_0035",
            "X_0036",
            "X_0042",
            "X_0048",
            "X_0082",
            "X_0090",
        ],
        "water-b": [
            "X_0001",
            "X_0002",
            "X_0005",
            "X_0006",
            "X_0011",
            "X_0018",
            "X_0083",
            "X_0087",
        ],
        "water-c": [
            "X_0002",
            "X_0010",
        ],
    }
    if source_file == "water-b":
        test_reps = ["1", "2", "3a", "3b", "4", "5", "6", "7",]
    else:
        test_reps = ["1", "2", "3", "4", "5", "6", "7", "8",]

    return strain_def[source_file], test_reps
