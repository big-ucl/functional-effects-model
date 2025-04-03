import pandas as pd
import time
import optuna
import pickle

from functools import partial

from helper import set_all_seeds
from utils import split_dataset, compute_metrics
from constants import PATH_TO_DATA_TRAIN, PATH_TO_FOLDS, alt_spec_features
from models_wrapper import RUMBoost, ResLogit, TasteNet


def objective(trial, model):
    """
    Optuna objective function for the hyperparameter search.
    """

    # load the data
    data = pd.read_csv(PATH_TO_DATA_TRAIN)

    with open(PATH_TO_FOLDS, "rb") as f:
        train_idx, val_idx = pickle.load(f)
        folds = zip(train_idx, val_idx)

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
        if col not in alt_spec_features
        and col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]

    if model == "RUMBoost":
        # parameters for RUMBoost
        args = {
            "model_type": "coral",
            "optim_interval": 20,
            "num_iterations": 3000,
            "early_stopping_rounds": 100,
            "verbose": 0,
            "verbose_interval": 100,
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, step=0.05),
            "device": "cuda",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
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
        model = RUMBoost(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=13,
            args=args,
        )
    elif args.model == "ResLogit":
        args = {
            "num_epochs": 200,
            "batch_size": trial.suggest_int("batch_size", 16, 256, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "patience": 20,
            "n_layers": trial.suggest_int("n_layers", 1, 32),
            "device": "cuda",
        }
        model = ResLogit(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=13,
            args=args,
        )
    elif args.model == "TasteNet":
        args = {
            "num_epochs": 200,
            "batch_size": trial.suggest_int("batch_size", 16, 256, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "patience": 20,
            "n_layers": trial.suggest_int("n_layers", 1, 32),
            "dropout": trial.suggest_float("dropout", 0.0, 0.9),
            "device": "cuda",
            "act_func": trial.suggest_categorical(
                "act_func", ["relu", "tanh", "sigmoid"]
            ),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "layer_sizes": [
                trial.suggest_categorical(
                    "layer_sizes",
                    [
                        [64],
                        [128],
                        [64, 64],
                        [128, 128],
                        [64, 128],
                        [128, 64],
                        [64, 128, 64],
                    ],
                ),
            ],
        }
        model = TasteNet(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=13,
            args=args,
        )

    avg_val_loss = 0.0
    avg_best_iter = 0.0
    k = len(folds)
    for train_idx, val_idx in folds:
        # split data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # build the dataloader
        model.build_dataloader(X_train, y_train, X_val, y_val)

        # fit the model
        _, best_val_loss = model.fit()
        avg_val_loss += best_val_loss
        avg_best_iter += model.best_iteration

    avg_best_iter /= k
    trial.set_user_attr("best_iteration", avg_best_iter)
    
    return avg_val_loss / k


if __name__ == "__main__":

    # set the random seed for reproducibility
    set_all_seeds(42)

    for model in ["RUMBoost", "ResLogit", "TasteNet"]:

        objective = partial(objective, model=model)

        study = optuna.create_study(direction="minimize")

        start_time = time.time()
        print(f"Starting hyperparameter search for {model}...")
        study.optimize(objective, n_trials=100)
        end_time = time.time()

        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        optimisation_time = end_time - start_time

        best_params["best_iteration"] = best_trial.user_attrs["best_iteration"]

        print(f"Best params: {best_params}")
        print(f"Best value: {best_value}")

        with open(f"results/{model}/best_params.pkl", "wb") as f:
            pickle.dump(best_params, f)

        with open(f"results/{model}/hyper_search_info.txt", "wb") as f:
            f.write(f"Best value: {best_value}\n")
            f.write(f"Optimisation time: {optimisation_time}\n")
