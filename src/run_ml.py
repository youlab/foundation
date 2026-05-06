if __name__ == "__main__":

    RUN_ADD_TO_COLLECTION = False
    RUN_UPLOAD_A7X = False
    RUN_UPLOAD_MNM = False
    RUN_UPLOAD_MCR = False
    RUN_UPLOAD_PR = False
    RUN_UPLOAD_VB = False
    RUN_TRAIN = True
    RUN_STUDIES_TO_DF = False
    RUN_OPTIMIZE = False

    if RUN_UPLOAD_A7X:
        print("Running RUN_UPLOAD_A7X")
        response = input("Are you sure you want to run this? This pushes to HuggingFace.[Y/n]")
        if response == "Y":
            from ml.upload_models.upload_a7x import upload_a7x
            upload_a7x()

    if RUN_UPLOAD_MNM:
        print("Running RUN_UPLOAD_MNM")
        response = input("Are you sure you want to run this? This pushes to HuggingFace.[Y/n]")
        if response == "Y":
            from ml.upload_models.upload_mnm import upload_mnm
            upload_mnm()

    if RUN_UPLOAD_MCR:
        print("Running RUN_UPLOAD_MCR")
        response = input("Are you sure you want to run this? This pushes to HuggingFace.[Y/n]")
        if response == "Y":
            from ml.upload_models.upload_mcr import upload_mcr
            upload_mcr()

    if RUN_UPLOAD_PR:
        print("Running RUN_UPLOAD_PR")
        response = input("Are you sure you want to run this? This pushes to HuggingFace.[Y/n]")
        if response == "Y":
            from ml.upload_models.upload_pr import upload_pr
            upload_pr()

    if RUN_UPLOAD_VB:
        print("Running RUN_UPLOAD_VB")
        response = input("Are you sure you want to run this? This pushes to HuggingFace.[Y/n]")
        if response == "Y":
            from ml.upload_models.upload_vb import upload_vb
            upload_vb()

    if RUN_ADD_TO_COLLECTION:
        print("Running RUN_ADD_TO_COLLECTION")
        response = input("Are you sure you want to run this? This pushes to HuggingFace.[Y/n]")
        if response == "Y":
            from ml.upload_models.add_to_collection import add_to_collection
            add_to_collection()

    if RUN_TRAIN:
        print("Running RUN_TRAIN")
        import sys
        from ml.train import main
        from config import (
            MODEL_TYPE,
            Z_DIM,
        )
        model_type = sys.argv[1]
        z_dim = int(sys.argv[2])
        data_category = sys.argv[3]
        if (
            model_type is None
        ) or (
            z_dim is None
        ) or (
            data_category is None
        ):
            print(f"Using default model type {MODEL_TYPE} and z_dim {Z_DIM} and data category all")
            model_type = MODEL_TYPE
            z_dim = Z_DIM
            data_category = "all"
        else:
            print(f"Using user-specified model type {model_type} and z_dim {z_dim} and data category {data_category}")

        main(
            model_type=model_type,
            z_dim=z_dim,
            config=None,
            data_category=data_category,
        )

    if RUN_STUDIES_TO_DF:
        print("Running RUN_STUDIES_TO_DF")
        from ml.studies_to_df import main
        main()

    if RUN_OPTIMIZE:
        print("Running RUN_OPTIMIZE")
        from ml.optimize import main
        main()
