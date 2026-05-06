if __name__ == "__main__":

    RUN_COMPILE = True
    RUN_NORMALIZE = True
    RUN_INTRINSIC_DIMENSION_ESTIMATION = False
    RUN_WRITE_DATA_SOURCES = False
    RUN_UPLOAD_DATA_EXPERIMENTAL = False
    RUN_UPLOAD_DATA_SIMULATION = False
    RUN_UPLOAD_DATA_PROCESSED = False
    RUN_ADD_TO_COLLECTION = False

    if RUN_NORMALIZE:
        print(f"\nRunning RUN_NORMALIZE")
        from data.normalize import main
        main()

    if RUN_COMPILE:
        print(f"\nRunning RUN_COMPILE")
        from data.compile import main
        main()

    if RUN_INTRINSIC_DIMENSION_ESTIMATION:
        print(f"Running RUN_INTRINSIC_DIMENSION_ESTIMATION")
        from data.intrinsic_dimension_estimation import main
        main()

    if RUN_WRITE_DATA_SOURCES:
        print(f"Running RUN_WRITE_DATA_SOUCES")
        from data.write_data_sources_table import main
        main()

    if RUN_UPLOAD_DATA_EXPERIMENTAL:
        print(f"Running RUN_UPLOAD_DATA_EXPERIMENTAL")
        response = input("Are you sure you want to upload data? [Y/n]")
        if response == "Y":
            from data.upload_data.upload_experimental import main
            main()

    if RUN_UPLOAD_DATA_SIMULATION:
        print(f"Running RUN_UPLOAD_DATA_SIMULATION")
        response = input("Are you sure you want to upload data? [Y/n]")
        if response == "Y":
            from data.upload_data.upload_simulation import main
            main()

    if RUN_UPLOAD_DATA_PROCESSED:
        print(f"Running RUN_UPLOAD_DATA_PROCESSED")
        response = input("Are you sure you want to upload data? [Y/n]")
        if response == "Y":
            from data.upload_data.upload_processed import main
            main()

    if RUN_ADD_TO_COLLECTION:
        print(f"Running RUN_ADD_TO_COLLECTION")
        response = input("Are you sure you want to run add to collection? [Y/n]")
        if response == "Y":
            from data.upload_data.add_to_collection import add_to_collection
            add_to_collection()
