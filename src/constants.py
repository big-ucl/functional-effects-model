PATH_TO_DATA = "../data/easySHARE/easySHARE_preprocessed.csv"
PATH_TO_DATA_TRAIN = "../data/easySHARE/easySHARE_preprocessed_train.csv"
PATH_TO_DATA_VAL = "../data/easySHARE/easySHARE_preprocessed_val.csv"
PATH_TO_DATA_TEST = "../data/easySHARE/easySHARE_preprocessed_test.csv"
PATH_TO_FOLDS = "../data/easySHARE/easySHARE_preprocessed_folds.pickle"

alt_spec_features = [
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
