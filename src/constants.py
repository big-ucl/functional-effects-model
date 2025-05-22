PATH_TO_DATA = {
    "easySHARE": "../data/easySHARE/easySHARE_preprocessed.csv",
    "SwissMetro": "",
}
PATH_TO_DATA_TRAIN = {
    "easySHARE": "../data/easySHARE/easySHARE_preprocessed_train.csv",
    "SwissMetro": "../data/SwissMetro/train.pkl",
}
PATH_TO_DATA_VAL = {
    "easySHARE": "../data/easySHARE/easySHARE_preprocessed_val.csv",
    "SwissMetro": "../data/SwissMetro/dev.pkl",
}
PATH_TO_DATA_TEST = {
    "easySHARE": "../data/easySHARE/easySHARE_preprocessed_test.csv",
    "SwissMetro": "../data/SwissMetro/test.pkl",
}
PATH_TO_FOLDS = {"easySHARE": "../data/easySHARE/easySHARE_folds.pkl", "SwissMetro": ""}

alt_spec_features = {
    "easySHARE": {
        0: [
            "chronic_mod",
            "nb_doctor_visits",
            "maxgrip",
            "daily_activities_index",
            "instrumental_activities_index",
            "mobilityind",
            "lgmuscle",
            "grossmotor",
            "finemotor",
            "recall_1",
            "recall_2",
            "bmi",
            # "sphus_excellent", normalised to 0
            "sphus_fair",
            "sphus_good",
            "sphus_poor",
            "sphus_very_good",
            "hospitalised_last_year_yes",
            # "hospitalised_last_year_no", normalised to 0
            "nursing_home_last_year_yes_permanently",
            "nursing_home_last_year_yes_temporarily",
            # "nursing_home_last_year_no", normalised to 0
        ]
    },
    "SwissMetro": {
        0: [
            "TRAIN_TT",
            "TRAIN_HE",
            "TRAIN_CO",
        ],
        1: [
            "SM_TT",
            "SM_HE",
            "SM_CO",
            "SM_SEATS",
        ],
        2: [
            "CAR_TT",
            "CAR_CO",
        ],
    },
}
