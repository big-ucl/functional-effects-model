# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import os
import argparse
import gc
import pickle
import optuna
import biogeme.database as db
import lightgbm as lgb

from functools import partial
from scipy.special import softmax
from rumboost.metrics import cross_entropy

from models_wrapper import RUMBoost, TasteNet, MixedEffect
from parser import parse_cmdline_args
from helper import set_all_seeds

# set seed for reproducibility
set_all_seeds(0)

n_alternatives = 4
num_observations = 100000
panel_factor = 10
features = [
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
]
alt_spec_features = {
    0: ["f4"],
    1: ["f5"],
    2: ["f6"],
    3: ["f7"],
}
socio_demo_chars = [
    "f0",
    "f1",
    "f2",
    "f3",
]
all_models = {
    "RUMBoost": RUMBoost,
    "TasteNet": TasteNet,
    # "MixedEffect": MixedEffect,
}


# Define the utility function
def utility_function(
    data: pd.DataFrame,
    with_noise: bool = False,
    with_intercept: bool = False,
    with_slopes: bool = False,
) -> np.ndarray:
    """
    Create the utility function for the synthetic dataset.

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    with_noise: bool
        Whether to add noise to the utility values
    with_intercept: bool
        Whether to include an intercept term
    with_slopes: bool
        Whether to include slope parameters

    Returns
    -------
    V: np.ndarray
        The utility values for each alternative.
    """
    # Extract the parameters
    V = np.zeros((data.shape[0], n_alternatives))

    if with_intercept:
        fct_intercepts = create_functional_intercepts(
            data.values, n_utility=n_alternatives, n_socio_dem=len(socio_demo_chars)
        )
    else:
        fct_intercepts = np.zeros((data.shape[0], n_alternatives))

    if with_slopes:
        coefficients = create_functional_slopes(
            data.values, n_utility=n_alternatives, n_socio_dem=len(socio_demo_chars)
        )
    else:
        coefficients = -np.ones((data.shape[0], n_alternatives))

    for i in range(n_alternatives):
        V[:, i] = (
            fct_intercepts[:, i]
            + coefficients[:, i] * data.values[:, i + len(socio_demo_chars)]
        )

    if with_noise:
        noise = generate_noise(0, 0.1, (data.shape[0], n_alternatives))
        V += noise

    return V


def generate_x(
    n: int, k: int, n_socio_dem: int = 0, panel_factor: int = 1
) -> np.ndarray:
    """
    Generate synthetic data.

    Parameters
    ----------
    n: int
        The total number of samples.
    k: int
        The total number of features.
    n_socio_dem: int
        The number of socio-demographic features.
    panel_factor: int
        The panel factor, i.e. the number of repeated trips per observation.

    Returns
    -------
    np.ndarray
        The generated synthetic data.
    """
    # socio-demographic variables
    if n_socio_dem > 0:
        n_s = int(n / panel_factor)
        x = np.random.uniform(0, 1, (n_s, n_socio_dem))
        x_socio_dem = x.repeat(panel_factor, axis=0)
    else:
        x_socio_dem = np.empty((n, 0))

    # alternative specific variables
    n_alt_spec = k - n_socio_dem

    x_alt_spec = np.random.uniform(0, 1, (n, n_alt_spec))

    return np.concatenate([x_socio_dem, x_alt_spec], axis=1)


def create_dataset() -> pd.DataFrame:
    """
    Create a pandas DataFrame from the synthetic data array.

    Returns
    -------
    pd.DataFrame
        The created DataFrame.
    """
    x_arr = generate_x(
        n=num_observations,
        k=len(features),
        n_socio_dem=len(socio_demo_chars),
        panel_factor=panel_factor,
    )

    x_arr_test = generate_x(
        n=int(0.2 * num_observations),
        k=len(features),
        n_socio_dem=len(socio_demo_chars),
        panel_factor=panel_factor,
    )

    data_train, data_test = (
        pd.DataFrame({features[i]: x_arr[:, i] for i in range(x_arr.shape[1])}),
        pd.DataFrame(
            {features[i]: x_arr_test[:, i] for i in range(x_arr_test.shape[1])}
        ),
    )

    return data_train, data_test


def create_functional_intercepts(
    x: np.ndarray, n_utility: int, n_socio_dem: int
) -> np.ndarray:
    """
    Create functional intercepts for a given number of utilities and features per utility.
    The functional intercepts are bounded by [0,1] and use all socio-demographic characteristics.
    This function assumes that the socio-demographic characteristics are in the first columns of the input array.

    Parameters
    ----------
    x: np.ndarray
        The input array containing the features.
    n_utility: int
        The number of utility functions to create.
    n_socio_dem: int
        The number of socio-demographic features.

    Returns
    -------
    np.ndarray
        The created functional intercepts.
    """
    effects = np.zeros((x.shape[0], n_utility))
    for i in range(n_utility):
        if i == 0:
            effects[:, i] = np.prod(np.exp(x[:, :n_socio_dem]), axis=1)
        elif i == 1:
            effects[:, i] = np.sum(x[:, :n_socio_dem], axis=1) ** 2
        elif i == 2:
            effects[:, i] = -np.log(np.prod(x[:, :n_socio_dem], axis=1))

        if i < n_utility - 1:
            effects[:, i] = effects[:, i] / effects[:, i].max()

    return effects


def create_functional_slopes(
    x: np.ndarray, n_utility: int, n_socio_dem: int
) -> np.ndarray:
    """
    Create functional slopes for a given number of utilities and features per utility.
    The functional slopes are bounded by [-1,0] and use all socio-demographic characteristics.
    This function assumes that the socio-demographic characteristics are in the first columns of the input array.

    Parameters
    ----------
    x: np.ndarray
        The input array containing the features.
    n_utility: int
        The number of utility functions to create.
    n_socio_dem: int
        The number of socio-demographic features.

    Returns
    -------
    np.ndarray
        The created functional slopes.
    """
    effects = np.zeros((x.shape[0], n_utility))
    for i in range(n_utility):
        if i == 0:
            effects[:, i] = -np.prod(np.exp(x[:, 0:2]), axis=1)
        elif i == 1:
            effects[:, i] = -(np.sum(x[:, 1:3], axis=1) ** 2)
        elif i == 2:
            effects[:, i] = np.log(np.prod(x[:, 0:3], axis=1))
        elif i == 3:
            effects[:, i] = -np.sqrt(np.sum(x[:, 1:n_socio_dem], axis=1))

        effects[:, i] = effects[:, i] / np.abs(effects[:, i]).max()

    return effects


def generate_noise(mean: float, sd: float, n: tuple[int, ...]) -> np.ndarray:
    """
    Generate noise from a Gumbel distribution.

    Parameters
    ----------
    mean: float
        The mean of the Gumbel distribution.
    sd: float
        The standard deviation of the Gumbel distribution.
    n: tuple
        The shape of the noise to generate.

    Returns
    -------
    noise: np.ndarray
        The generated noise.
    """
    return np.random.gumbel(loc=mean, scale=sd, size=n)


def compute_prob(V: np.ndarray) -> np.ndarray:
    """
    Compute the probabilities for each alternative using the softmax function.

    Parameters
    ----------
    V: np.ndarray
        The utility values for each alternative.

    Returns
    -------
    probs: np.ndarray
        The probabilities for each alternative.
    """

    return softmax(V, axis=1)


def generate_labels(probs: np.ndarray) -> np.ndarray:
    """
    Generate labels based on the probabilities.

    Parameters
    ----------
    probs: np.ndarray
        The probabilities for each alternative and each observation.

    Returns
    -------
    labels: np.ndarray
        The generated labels for each alternative and each observation.
    """
    labels = [
        np.random.choice(range(n_alternatives), p=probs[i])
        for i in range(probs.shape[0])
    ]
    return np.array(labels)


def add_simulated_choices(
    data: pd.DataFrame,
    with_noise: bool = False,
    with_intercept: bool = True,
    with_slopes: bool = True,
) -> pd.DataFrame:
    """
    Add simulated choices to the data based on the utility function.

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    with_noise: bool
        Whether to add noise to the utility values
    with_intercept: bool
        Whether to include an intercept term in the utility function
    with_slopes: bool
        Whether to include slopes in the utility function

    Returns
    -------
    data: pd.DataFrame
        Data with the simulated choices added.
    """
    V = utility_function(
        data,
        with_noise=with_noise,
        with_intercept=with_intercept,
        with_slopes=with_slopes,
    )
    probs = compute_prob(V)
    data["choice"] = generate_labels(probs)
    return data


def gather_functional_intercepts(
    data: pd.DataFrame,
    model: RUMBoost | TasteNet | MixedEffect,
    socio_demo_characts: list[str] = socio_demo_chars,
    n_classes: int = n_alternatives,
    alt_normalised: int = 0,
    on_train_set: bool = True,
    functional_intercept: bool = True,
    functional_params: bool = False,
) -> np.ndarray:
    """
    Gather the learnt functional intercepts for the given model.

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    model: RUMBoost or TasteNet
        The model used for the synthetic experiment.
    socio_demo_characts: list[str], optional (default: socio_demo_chars)
        The socio-demographic characteristics used for the functional intercepts.
    n_classes: int, optional (default: n_alternatives)
        The number of alternatives (classes) in the model.
    alt_normalised: int, optional (default: 0)
        The alternative index for the normalised functional intercepts.
    on_train_set: bool, optional (default: True)
        Whether to compute the functional intercepts on the training set or not.
        Only used for MixedEffect model.
    functional_intercept: bool, optional
        If the model uses functional intercepts.
    functional_params: bool, optional
        If the model uses functional parameters.

    Returns
    -------
    functional_intercepts: np.ndarray
        The functional intercepts for the given features.
    """

    if isinstance(model, RUMBoost):

        if not functional_params:
            dummy_array = np.zeros((1,)).reshape(-1, 1)
            asc = np.zeros(
                (
                    1,
                    n_classes,
                )
            )
            for i, booster in enumerate(model.model.boosters[:n_classes]):
                asc[0, i] = booster.predict(dummy_array, raw_score=True)

            dummy_df = pd.DataFrame(
                {
                    "f0": [0],
                    "f1": [0],
                    "f2": [0],
                    "f3": [0],
                    "f4": [0],
                    "f5": [0],
                    "f6": [0],
                    "f7": [0],
                }
            )

            
        if functional_intercept:
            rumboost_predictor = model.model.boosters[-n_classes:]
            fct_intercepts = np.zeros((data.shape[0], n_classes))
            for i, predictor in enumerate(rumboost_predictor):
                fct_intercepts[:, i] = predictor.predict(
                    data[socio_demo_characts], raw_score=True
                )
            if not functional_params:
                fct_intercepts += asc
        else:
            dummy_array = pd.DataFrame(
                {
                    "f0": [0],
                    "f1": [0],
                    "f2": [0],
                    "f3": [0],
                    "f4": [0],
                    "f5": [0],
                    "f6": [0],
                    "f7": [0],
                }
            )
            fct_intercepts, _, _ = model.predict(dummy_array)
            if not functional_params:
                fct_intercepts += asc
            fct_intercepts = fct_intercepts.reshape(1, -1).repeat(data.shape[0], axis=0)

        if functional_params:
            rumboost_predictor = model.model.boosters[:4]
            fct_slopes = np.zeros((data.shape[0], 4))
            for i, predictor in enumerate(rumboost_predictor):
                fct_slopes[:, i] = -np.maximum(
                    -predictor.predict(data[socio_demo_characts], raw_score=True), 0
                )
        else:
            rumboost_predictor = model.model.boosters[:4]
            fct_slopes = np.zeros((data.shape[0], 4))
            for i, predictor in enumerate(rumboost_predictor):
                preds = (
                    predictor.predict(
                        data.values[:, i + n_classes].reshape(-1, 1), raw_score=True
                    )
                )
                fct_slopes[:, i] = (preds - asc[:,i]) / data.values[:, i + n_classes]
    elif isinstance(model, TasteNet):
        if functional_intercept:
            tastenet_predictor = model.model.params_module
            sdc_tensor = (
                torch.from_numpy(data[socio_demo_characts].values)
                .to(torch.device("cuda"))
                .to(torch.float32)
            )
            fct_intercepts = (
                tastenet_predictor(sdc_tensor).detach().cpu().numpy()[:, -n_classes:]
            )
        else:
            fct_intercepts = (
                model.model.util_module.intercept.view(1, n_classes)
                .detach()
                .cpu()
                .numpy()
                .reshape(1, -1)
                .repeat(data.shape[0], axis=0)
            )

        if functional_params:
            tastenet_predictor = model.model.params_module
            sdc_tensor = (
                torch.from_numpy(data[socio_demo_characts].values)
                .to(torch.device("cuda"))
                .to(torch.float32)
            )
            fct_slopes = -np.maximum(
                -tastenet_predictor(sdc_tensor).detach().cpu().numpy()[:, :4], 0
            )
        else:
            fct_slopes = np.array([])
            for param in model.model.util_module.mnl.mnl.parameters():
                fct_slopes = np.concatenate(
                    [fct_slopes, param.detach().cpu().numpy()[0]]
                )
            fct_slopes = fct_slopes.reshape(1, -1).repeat(data.shape[0], axis=0)


    fct_effects = np.concatenate([fct_slopes, fct_intercepts], axis=1)
    if functional_intercept:
        fct_effects[:, -n_classes:] = fct_effects[:, -n_classes:] - fct_effects[
            :, -n_classes + alt_normalised
        ].reshape(-1, 1)
    return fct_effects


def l1_distance(
    true_fct_intercept: np.ndarray, learnt_fct_intercept: np.ndarray
) -> float:
    """
    Compute the L1 distance between the true and learnt functional intercepts.

    Parameters
    ----------
    true_fct_intercept: np.ndarray
        The true functional intercepts.
    learnt_fct_intercept: np.ndarray
        The learnt functional intercepts.

    Returns
    -------
    l1_distance: float
        The L1 distance between the true and learnt functional intercepts.
    """
    return np.sum(np.abs(true_fct_intercept - learnt_fct_intercept), axis=0)

def mape(
    true_fct_intercept: np.ndarray, learnt_fct_intercept: np.ndarray
) -> float:
    """
    Compute the L1 distance between the true and learnt functional intercepts.

    Parameters
    ----------
    true_fct_intercept: np.ndarray
        The true functional intercepts.
    learnt_fct_intercept: np.ndarray
        The learnt functional intercepts.

    Returns
    -------
    l1_distance: float
        The L1 distance between the true and learnt functional intercepts.
    """
    return np.sum(np.abs((true_fct_intercept - learnt_fct_intercept)/(true_fct_intercept + 1e-6)), axis=0)


def run_experiment(args: argparse.Namespace) -> None:
    """
    Run the synthetic experiment with the given arguments.

    Parameters
    ----------
    args: argparse.Namespace
        The command line arguments parsed by the parser.
    """
    # reset all seeds at the beginning of each experiment for reproducibility
    set_all_seeds(0)

    args.outpath = f"results/synthetic_withint{args.with_intercept}_withslopes{args.with_slopes}/{args.model}/"
    os.makedirs(args.outpath, exist_ok=True)

    # load data
    data_train, data_test = create_dataset()

    # create synthetic utility values and choices
    data_train = add_simulated_choices(
        data_train,
        with_noise=True,
        with_intercept=args.with_intercept,
        with_slopes=args.with_slopes,
    )
    data_test = add_simulated_choices(
        data_test,
        with_noise=True,
        with_intercept=args.with_intercept,
        with_slopes=args.with_slopes,
    )

    X_train, y_train = data_train[features], data_train["choice"]
    X_val, y_val = None, None
    X_test, y_test = data_test[features], data_test["choice"]

    if args.with_intercept and args.with_slopes:
        fct_intercepts = create_functional_intercepts(
            data_train.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_intercepts_test = create_functional_intercepts(
            data_test.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_slopes = create_functional_slopes(
            data_train.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_slopes_test = create_functional_slopes(
            data_test.values,
            n_alternatives,
            len(socio_demo_chars),
        )

        fct_effects = np.concatenate([fct_slopes, fct_intercepts], axis=1)
        fct_effects_test = np.concatenate(
            [fct_slopes_test, fct_intercepts_test], axis=1
        )
    elif args.with_slopes:
        fct_slopes = create_functional_slopes(
            data_train.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_slopes_test = create_functional_slopes(
            data_test.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_effects = np.concatenate(
            [fct_slopes, np.zeros((X_train.shape[0], n_alternatives))], axis=1
        )
        fct_effects_test = np.concatenate(
            [fct_slopes_test, np.zeros((X_test.shape[0], n_alternatives))], axis=1
        )
    elif args.with_intercept:
        fct_intercepts = create_functional_intercepts(
            data_train.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_intercepts_test = create_functional_intercepts(
            data_test.values,
            n_alternatives,
            len(socio_demo_chars),
        )
        fct_effects = np.concatenate(
            [-np.ones((X_train.shape[0], 4)), fct_intercepts], axis=1
        )
        fct_effects_test = np.concatenate(
            [-np.ones((X_test.shape[0], 4)), fct_intercepts_test], axis=1
        )
    else:
        fct_effects = np.concatenate(
            [
                -np.ones((X_train.shape[0], 4)),
                np.zeros((X_train.shape[0], n_alternatives)),
            ],
            axis=1,
        )
        fct_effects_test = np.concatenate(
            [
                -np.ones((X_test.shape[0], 4)),
                np.zeros((X_test.shape[0], n_alternatives)),
            ],
            axis=1,
        )

    # define instances of the models
    if args.model == "RUMBoost":
        model = RUMBoost(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=n_alternatives,
            args=args,
        )
        save_path = (
            args.outpath
            + f"model_fi{args.functional_intercept}_fp{args.functional_params}.json"
        )
    elif args.model == "TasteNet":
        model = TasteNet(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=n_alternatives,
            num_latent_vals=None,
            args=args,
        )
        save_path = (
            args.outpath
            + f"model_fi{args.functional_intercept}_fp{args.functional_params}.pth"
        )

    # build dataloader
    model.build_dataloader(X_train, y_train, X_val, y_val)

    # fit model
    start_time = time.time()
    best_train_loss, best_val_loss = model.fit(save_path=save_path)
    end_time = time.time()

    # predict on the test set

    preds, _, _ = model.predict(X_test)
    loss_test = cross_entropy(preds, y_test)

    # get learnt functional intercepts
    learnt_fct_effects = gather_functional_intercepts(
        X_train,
        model,
        socio_demo_chars,
        n_alternatives,
        alt_normalised=3,
        on_train_set=True,
        functional_intercept=args.functional_intercept,
        functional_params=args.functional_params,
    )
    learnt_fct_effects_test = gather_functional_intercepts(
        X_test,
        model,
        socio_demo_chars,
        n_alternatives,
        alt_normalised=3,
        on_train_set=False,
        functional_intercept=args.functional_intercept,
        functional_params=args.functional_params,
    )

    l1_distance_train = l1_distance(
        fct_effects,
        learnt_fct_effects,
    )
    l1_distance_test = l1_distance(
        fct_effects_test,
        learnt_fct_effects_test,
    )

    avg_l1_dist = np.mean(l1_distance_train / X_train.shape[0])
    avg_l1_dist_test = np.mean(l1_distance_test / X_test.shape[0])

    mape_train = mape(
        fct_effects,
        learnt_fct_effects,
    )
    mape_test = mape(
        fct_effects_test,
        learnt_fct_effects_test,
    )

    avg_l1_dist = np.mean(l1_distance_train / X_train.shape[0])
    avg_l1_dist_test = np.mean(l1_distance_test / X_test.shape[0])

    avg_mape_train = np.mean(mape_train / X_train.shape[0])
    avg_mape_test =  np.mean(mape_test / X_test.shape[0])

    print(f"Best Train Loss: {best_train_loss}, Best Val Loss: {best_val_loss}")
    print(f"Test Loss: {loss_test}")
    print(f"L1 distance per effects: {l1_distance_train}")
    print(f"L1 distance per effects on the test set: {l1_distance_test}")
    print(f"Average L1 distance per effect and observation: {avg_l1_dist}")
    print(
        f"Average L1 distance per effect and observation on the test set: {avg_l1_dist_test}"
    )

    results_dict = {
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "loss_test": loss_test,
        "train_time": end_time - start_time,
        "avg_l1_dist": avg_l1_dist,
        "avg_l1_dist_test": avg_l1_dist_test,
        "mape": avg_mape_train,
        "mape_test": avg_mape_test,
    }

    if args.save_model:
        # save the results
        pd.DataFrame(results_dict, index=[0]).to_csv(
            args.outpath
            + f"results_dict_fi{args.functional_intercept}_fp{args.functional_params}.csv"
        )

        model.save_model(save_path)


def hyperparameter_search(
    model: str = "RUMBoost",
    with_intercept: bool = True,
    with_slopes: bool = True,
    functional_params: bool = False,
    functional_intercept: bool = True,
) -> None:
    """
    Perform hyperparameter search for the models.
    This function is not implemented yet.

    Parameters
    ----------
    model : str
        The model to train. Can be "RUMBoost" or "TasteNet".
    with_intercept : bool
        Whether to use functional intercept.
    with_slopes : bool
        Whether to use functional slopes.
    functional_params : bool
        Whether to use functional parameters.
    functional_intercept : bool
        Whether to use functional intercept.
    """
    # reset all seeds at the beginning of each experiment for reproducibility
    set_all_seeds(0)

    # load data
    data_train, _ = create_dataset()

    # create synthetic utility values and choices
    data_train = add_simulated_choices(
        data_train,
        with_noise=True,
        with_intercept=with_intercept,
        with_slopes=with_slopes,
    )

    X, y = data_train[features], data_train["choice"]

    # ensuring a split that does not have individuals in both train and validation sets
    train_indices, val_indices = (
        np.arange(int(X.shape[0] * 0.8)),
        np.arange(int(X.shape[0] * 0.8), X.shape[0]),
    )

    X_train, y_train = X.iloc[train_indices].copy(), y.iloc[train_indices].copy()
    X_val, y_val = X.iloc[val_indices].copy(), y.iloc[val_indices].copy()

    def objective(
        trial: optuna.Trial, model: str, func_int: bool, func_params: bool
    ) -> float:
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

        Returns
        -------
        float
            The average validation loss over the folds.
        """

        # default args
        args = parse_cmdline_args()

        num_classes = n_alternatives

        if model == "RUMBoost":
            # parameters for RUMBoost
            dict_args = {
                "dataset": "synthetic",
                "model_type": "",
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
                alt_spec_features=alt_spec_features,
                socio_demo_chars=socio_demo_chars,
                num_classes=num_classes,
                args=args,
            )
        elif model == "TasteNet":
            dict_args = {
                "dataset": "synthetic",
                "num_epochs": 200,
                "functional_intercept": func_int,
                "functional_params": func_params,
                "verbose": 0,
                "batch_size": trial.suggest_int("batch_size", 256, 512, step=256),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
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
                alt_spec_features=alt_spec_features,
                socio_demo_chars=socio_demo_chars,
                num_classes=num_classes,
                num_latent_vals=None,
                args=args,
            )

        # build the dataloader
        model.build_dataloader(X_train, y_train, X_val, y_val)

        # fit the model
        _, best_val_loss = model.fit()
        best_iter = model.best_iteration

        trial.set_user_attr("best_iteration", best_iter)

        del model

        gc.collect()
        torch.cuda.empty_cache()

        return best_val_loss

    func_int = functional_intercept
    func_params = functional_params

    objective = partial(
        objective,
        model=model,
        func_int=func_int,
        func_params=func_params,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///optuna_study_fi{func_int}_fp{func_params}_model{model}_synthetic_wi{with_intercept}_ws{with_slopes}.db",
        load_if_exists=True,
    )

    start_time = time.time()
    study.optimize(objective, n_trials=100, n_jobs=1)
    end_time = time.time()

    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    optimisation_time = end_time - start_time

    best_params["best_iteration"] = best_trial.user_attrs["best_iteration"]

    print(f"Best params: {best_params}")
    print(f"Best value: {best_value}")

    dataset = f"synthetic_withint{with_intercept}_withslopes{with_slopes}"
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


if __name__ == "__main__":
    for with_intercept in [False, True]:  # , False]:#,
        for with_slopes in [False, True]:  # , False]:#,
            for functional_params in [False, True]:  # , False]:#,
                for functional_intercept in [False, True]:  # , False]:#,
                    for model in all_models.keys():
                        # run hyperparameter search
                        # hyperparameter_search(model=model)
                        # if os.path.exists(
                        #     f"results/synthetic_withint{with_intercept}_withslopes{with_slopes}/{model}/results_dict_fi{functional_intercept}_fp{functional_params}.csv"
                        # ):
                        #     print(
                        #         f"Results already exist for {model} with functional_intercept={functional_intercept} and functional_params={functional_params}. Skipping."
                        #     )
                        #     continue

                        # load the optimal hyperparameters for the model
                        args = parse_cmdline_args()
                        try:
                            opt_hyperparams_path = f"results/synthetic_withint{with_intercept}_withslopes{with_slopes}/{model}/best_params_fi{functional_intercept}_fp{functional_params}.pkl"
                            with open(opt_hyperparams_path, "rb") as f:
                                optimal_hyperparams = pickle.load(f)
                                if "layer_sizes" in optimal_hyperparams:
                                    optimal_hyperparams["layer_sizes"] = [
                                        int(size)
                                        for size in optimal_hyperparams[
                                            "layer_sizes"
                                        ].split(",")
                                    ]
                                if "learning_rate" not in optimal_hyperparams:
                                    optimal_hyperparams["learning_rate"] = 1
                                args.__dict__.update(optimal_hyperparams)
                        except FileNotFoundError:
                            print(
                                f"Optimal hyperparameters not found for {model}. Running hyperparameter search."
                            )
                            hyperparameter_search(
                                model=model,
                                with_intercept=with_intercept,
                                with_slopes=with_slopes,
                                functional_params=functional_params,
                                functional_intercept=functional_intercept,
                            )
                            opt_hyperparams_path = f"results/synthetic_withint{with_intercept}_withslopes{with_slopes}/{model}/best_params_fi{functional_intercept}_fp{functional_params}.pkl"
                            with open(opt_hyperparams_path, "rb") as f:
                                optimal_hyperparams = pickle.load(f)
                                if "layer_sizes" in optimal_hyperparams:
                                    optimal_hyperparams["layer_sizes"] = [
                                        int(size)
                                        for size in optimal_hyperparams[
                                            "layer_sizes"
                                        ].split(",")
                                    ]
                                if "learning_rate" not in optimal_hyperparams:
                                    optimal_hyperparams["learning_rate"] = 1
                                args.__dict__.update(optimal_hyperparams)
                        args.functional_intercept = functional_intercept
                        args.functional_params = functional_params
                        args.save_model = False 
                        args.model = model
                        args.dataset = "synthetic"
                        args.with_intercept = with_intercept
                        args.with_slopes = with_slopes
                        if model == "RUMBoost":
                            args.num_iterations = int(args.best_iteration)
                        elif model == "TasteNet":
                            args.num_epochs = int(args.best_iteration)
                        args.early_stopping_rounds = None
                        run_experiment(args)

                        gc.collect()
                        torch.cuda.empty_cache()
