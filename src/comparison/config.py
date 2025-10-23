from itertools import product

NOW_TEXT = "2025-02-28T09:00"


def get_params(append_32=True):
    learning_rates = [1e-2, 1e-3, 1e-4,]

    dim_heads = [
        (24, 24,),
        (24, 12,),
        (24, 8,),
        (24, 6,),
        (24, 4,),
        (24, 2,),
        (20, 20,),
        (20, 10,),
        (20, 4,),
        (20, 2,),
        (16, 16,),
        (16, 8,),
        (16, 4,),
        (16, 2,),
        (12, 12,),
        (12, 4,),
        (8, 8,),
        (8, 4,),
        (8, 2,),
        (6, 6,),
        (6, 2,),
        (4, 4,),
        (4, 2,),
        (2, 2,),
    ]

    n_layers = [
        8, 12, 16,
    ]

    use_random_mask = [False,]
    loss_fcn = ["full_curve",]

    params = list(product(learning_rates, dim_heads, n_layers, use_random_mask, loss_fcn,),)
    if not append_32:
        return params

    dim_heads = [
        (32, 32,),
        (32, 16,),
        (32, 8,),
        (32, 4,),
        (32, 2,),
    ]

    params.extend(list(product(learning_rates, dim_heads, n_layers, use_random_mask, loss_fcn,),))

    return params
