import pandas as pd
import time
import optuna
import pickle
import gc
import torch
import os

from sklearn.preprocessing import MinMaxScaler

from functools import partial

from helper import set_all_seeds
from utils import pkl_to_df
from constants import (
    PATH_TO_DATA,
    PATH_TO_DATA_TRAIN,
    PATH_TO_DATA_VAL,
    PATH_TO_FOLDS,
    alt_spec_features,
)
from models_wrapper import RUMBoost, TasteNet
from parser import parse_cmdline_args

from rumboost.datasets import load_preprocess_LPMC


def objective(trial, model, func_int, func_params, dataset):
    """
    Optuna objective function for the hyperparameter search.

    Parameters
    ----------
    trial : optuna.Trial
        The current trial object.
    model : str
        The model to train.
    func_int : bool
        Whether to use functional intercept.
    func_params : bool
        Whether to use functional parameters.
    dataset : str
        The dataset to use.
    """

    all_alt_spec_features = []
    for _, value in alt_spec_features[dataset].items():
        all_alt_spec_features.extend(value)

    # load the data
    if dataset == "SwissMetro":
        data = pkl_to_df(PATH_TO_DATA_TRAIN[dataset])
        data_val = pkl_to_df(PATH_TO_DATA_VAL[dataset])

        features = [col for col in data.columns if col not in ["CHOICE"]]
        target = "CHOICE"

        X_train, y_train = data[features], data[target]
        X_val, y_val = data_val[features], data_val[target]

        socio_demo_chars = [
            col
            for col in data.columns
            if col not in all_alt_spec_features and col not in ["CHOICE"]
        ]
        folds = [("dummy", "dummy")]
        num_classes = 3
    elif dataset == "easySHARE":
        data = pd.read_csv(PATH_TO_DATA_TRAIN[dataset])

        with open(PATH_TO_FOLDS[dataset], "rb") as f:
            folds = pickle.load(f)

        features = [
            col
            for col in data.columns
            if col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
        ]
        target = "depression_scale"

        X, y = data[features], data[target]

        socio_demo_chars = [
            col
            for col in data.columns
            if col not in all_alt_spec_features
            and col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
        ]
        num_classes = 13
    elif dataset == "LPMC":
        data_train, _, folds = load_preprocess_LPMC(PATH_TO_DATA[dataset])
        features = [col for col in data_train.columns if col not in ["choice"]]
        target = "choice"

        X, y = data_train[features], data_train[target]

        socio_demo_chars = [
            col
            for col in data_train.columns
            if col not in all_alt_spec_features and col not in ["choice"]
        ]
        num_classes = 4

    # default args
    args = parse_cmdline_args()



    if model == "RUMBoost":
        # parameters for RUMBoost
        dict_args = {
            "dataset": dataset,
            "model_type": "coral" if dataset == "easySHARE" else "",
            "optim_interval": 20,
            "num_iterations": 3000,
            "early_stopping_rounds": 100,
            "verbose": 0,
            "functional_intercept": func_int,
            "functional_params": func_params,
            "learning_rate": 1,  # is modified in the model, divided by num of updated boosters or 0.1
            "device": "cuda",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 200),
            "max_bin": trial.suggest_int("max_bin", 64, 511),
            "min_sum_hessian_in_leaf": trial.suggest_float(
                "min_sum_hessian_in_leaf", 1e-8, 10.0, log=True
            ),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 1e-8, 10.0, log=True
            ),
        }
        args.__dict__.update(dict_args)
        model = RUMBoost(
            alt_spec_features=alt_spec_features[dataset],
            socio_demo_chars=socio_demo_chars,
            num_classes=num_classes,
            args=args,
        )
    elif model == "TasteNet":
        dict_args = {
            "dataset": dataset,
            "num_epochs": 200,
            "functional_intercept": func_int,
            "functional_params": func_params,
            "verbose": 0,
            "batch_size": trial.suggest_int("batch_size", 256, 512, step=256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "patience": 10,
            "dropout": trial.suggest_float("dropout", 0.0, 0.9),
            "device": "cuda",
            "act_func": trial.suggest_categorical(
                "act_func", ["relu", "tanh", "sigmoid"]
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1, log=True),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "layer_sizes": [
                trial.suggest_categorical(
                    "layer_sizes",
                    [
                        "32",
                        "64",
                        "128",
                        "32, 32",
                        "64, 64",
                        "128, 128",
                        "64, 128",
                        "128, 64",
                        "64, 128, 64",
                    ],
                ),
            ],
        }
        dict_args["layer_sizes"] = [
            int(size) for size in dict_args["layer_sizes"][0].split(", ")
        ]
        args.__dict__.update(dict_args)
        model = TasteNet(
            alt_spec_features=alt_spec_features[dataset],
            socio_demo_chars=socio_demo_chars,
            num_classes=num_classes,
            num_latent_vals=1 if dataset == "easySHARE" else None,
            args=args,
        )

    avg_val_loss = 0.0
    avg_best_iter = 0.0
    k = 1
    for i, (train_idx, val_idx) in enumerate(folds):
        if i > 0:
            continue  # can only do 1 fold because of computational time
        # split data
        if dataset in ["easySHARE", "LPMC"]:
            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

        # scale the features
        scaler = MinMaxScaler()
        X_train[all_alt_spec_features + socio_demo_chars] = scaler.fit_transform(
            X_train[all_alt_spec_features + socio_demo_chars]
        )
        X_val[all_alt_spec_features + socio_demo_chars] = scaler.transform(
            X_val[all_alt_spec_features + socio_demo_chars]
        )

        # build the dataloader
        model.build_dataloader(X_train, y_train, X_val, y_val)

        # fit the model
        _, best_val_loss = model.fit()
        avg_val_loss += best_val_loss
        avg_best_iter += model.best_iteration

    avg_best_iter /= k
    trial.set_user_attr("best_iteration", avg_best_iter)

    del model

    gc.collect()
    torch.cuda.empty_cache()

    return avg_val_loss / k


if __name__ == "__main__":

    # set the random seed for reproducibility
    set_all_seeds(42)
    for dataset in ["LPMC", "SwissMetro", "easySHARE"]:
        for model in ["RUMBoost"]:#"TasteNet"]:#,
            for func_int in [False]: #, False]:#,
                for func_params in [False]: #, False]:

                    objective = partial(
                        objective,
                        model=model,
                        func_int=func_int,
                        func_params=func_params,
                        dataset=dataset,
                    )

                    study = optuna.create_study(direction="minimize")

                    start_time = time.time()
                    print(
                        f"Starting hyperparameter search on dataset {dataset} for {model} with func params {func_params} and with func intercept {func_int}..."
                    )
                    study.optimize(objective, n_trials=100, n_jobs=1)
                    end_time = time.time()

                    best_params = study.best_params
                    best_value = study.best_value
                    best_trial = study.best_trial
                    optimisation_time = end_time - start_time

                    best_params["best_iteration"] = best_trial.user_attrs[
                        "best_iteration"
                    ]

                    print(f"Best params: {best_params}")
                    print(f"Best value: {best_value}")

                    path = f"results/{dataset}/{model}/"
                    # create the directory if it doesn't exist
                    os.makedirs(path, exist_ok=True)

                    with open(
                        f"results/{dataset}/{model}/best_params_fi{func_int}_fp{func_params}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(best_params, f)

                    with open(
                        f"results/{dataset}/{model}/hyper_search_info_fi{func_int}_fp{func_params}.txt",
                        "w",
                    ) as f:
                        f.write(f"Best value: {best_value}\n")
                        f.write(f"Optimisation time: {optimisation_time}\n")
