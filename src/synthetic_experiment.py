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
    "MixedEffect": MixedEffect,
}
coefficients = [-1, -1, -1, -1]


# Define the utility function
def utility_function(data: pd.DataFrame, with_noise: bool = False) -> np.ndarray:
    """
    Create the utility function for the synthetic dataset.

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    with_noise: bool
        Whether to add noise to the utility values

    Returns
    -------
    V: np.ndarray
        The utility values for each alternative.
    """
    # Extract the parameters
    V = np.zeros((data.shape[0], n_alternatives))

    fct_effects = create_functional_effects(
        data.values, n_utility=n_alternatives, n_socio_dem=len(socio_demo_chars)
    )

    for i in range(n_alternatives):
        V[:, i] = (
            fct_effects[:, i]
            + coefficients[i] * data.values[:, i + len(socio_demo_chars)]
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

    data_train, data_test = pd.DataFrame(
        {features[i]: x_arr[:, i] for i in range(x_arr.shape[1])}
    ), pd.DataFrame({features[i]: x_arr_test[:, i] for i in range(x_arr_test.shape[1])})

    return data_train, data_test


def create_functional_effects(
    x: np.ndarray, n_utility: int, n_socio_dem: int
) -> np.ndarray:
    """
    Create functional effects for a given number of utilities and features per utility.
    The functional effects are bounded by [0,1] and use all socio-demographic characteristics.
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
        The created functional effects.
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


def add_simulated_choices(data: pd.DataFrame, with_noise: bool = False) -> pd.DataFrame:
    """
    Add simulated choices to the data based on the utility function.

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    with_noise: bool
        Whether to add noise to the utility values

    Returns
    -------
    data: pd.DataFrame
        Data with the simulated choices added.
    """
    V = utility_function(data, with_noise=with_noise)
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

    Returns
    -------
    functional_intercepts: np.ndarray
        The functional intercepts for the given features.
    """
    if isinstance(model, RUMBoost):
        rumboost_predictor = model.model.boosters[-n_classes:]
        fct_intercept = np.zeros((data.shape[0], n_classes))
        for i, predictor in enumerate(rumboost_predictor):
            fct_intercept[:, i] = predictor.predict(
                data[socio_demo_characts], raw_score=True
            )
        dummy_array = np.zeros((1,)).reshape(-1, 1)
        asc = np.zeros(
            (
                1,
                n_classes,
            )
        )
        for i, booster in enumerate(model.model.boosters[:-n_classes]):
            asc[0, i] = booster.predict(dummy_array, raw_score=True)
        fct_intercept = fct_intercept + asc
    elif isinstance(model, TasteNet):
        tastenet_predictor = model.model.params_module
        # already computing the functional values as they are outputted all at once
        sdc_tensor = (
            torch.from_numpy(data[socio_demo_characts].values)
            .to(torch.device("cuda"))
            .to(torch.float32)
        )
        fct_intercept = tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()
    else:
        fct_intercept = model.get_individual_parameters(on_train_set=on_train_set)
        if on_train_set:
            fct_intercept = np.repeat(fct_intercept, panel_factor, axis=0)
        else:
            fct_intercept = np.repeat(fct_intercept.reshape(1, -1), num_observations * 0.2, axis=0)
    return fct_intercept - fct_intercept[:, alt_normalised].reshape(-1, 1)


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

    args.outpath = f"results/synthetic/{args.model}/"
    os.makedirs(args.outpath, exist_ok=True)

    # load data
    data_train, data_test = create_dataset()

    # create synthetic utility values and choices
    data_train = add_simulated_choices(
        data_train,
        with_noise=True,
    )
    data_test = add_simulated_choices(
        data_test,
        with_noise=True,
    )

    X_train, y_train = data_train[features], data_train["choice"]
    X_val, y_val = None, None
    X_test, y_test = data_test[features], data_test["choice"]

    # create synthetic "ground truth" functional intercepts
    fct_intercept = create_functional_effects(
        data_train.values,
        n_alternatives,
        len(socio_demo_chars),
    )
    fct_intercept_test = create_functional_effects(
        data_test.values,
        n_alternatives,
        len(socio_demo_chars),
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
    elif args.model == "MixedEffect":
        model = MixedEffect(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=n_alternatives,
        )
        X_train["ID"] = np.repeat(
            np.arange(int(X_train.shape[0] / panel_factor)), panel_factor
        )
        X_test["ID"] = np.repeat(
            np.arange(int(X_test.shape[0] / panel_factor)), panel_factor
        )

        save_path = (
            args.outpath
            + f"model_fi{args.functional_intercept}_fp{args.functional_params}.yaml"
        )

    # build dataloader
    model.build_dataloader(X_train, y_train, X_val, y_val)

    # fit model
    start_time = time.time()
    best_train_loss, best_val_loss = model.fit(save_path=save_path)
    end_time = time.time()

    # predict on the test set

    if args.model == "MixedEffect":
        loss_test, _, _ = model.predict(X_test, y_test=y_test)
    else:
        preds, _, _ = model.predict(X_test)
        loss_test = cross_entropy(preds, y_test)

    # get learnt functional intercepts
    learnt_fct_intercepts = gather_functional_intercepts(
        data_train,
        model,
        socio_demo_chars,
        n_alternatives,
        alt_normalised=3,
        on_train_set=True,
    )
    learnt_fct_intercepts_test = gather_functional_intercepts(
        data_test,
        model,
        socio_demo_chars,
        n_alternatives,
        alt_normalised=3,
        on_train_set=False,
    )

    # compute L1 distance between true and learnt functional intercepts
    l1_distance_train = l1_distance(
        fct_intercept,
        learnt_fct_intercepts,
    )
    l1_distance_test = l1_distance(
        fct_intercept_test,
        learnt_fct_intercepts_test,
    )

    mean_fct_intercept = np.mean(fct_intercept, axis=0)
    mean_learnt_fct_intercept = np.mean(learnt_fct_intercepts, axis=0)
    mean_fct_intercept_test = np.mean(fct_intercept_test, axis=0)
    mean_learnt_fct_intercept_test = np.mean(learnt_fct_intercepts_test, axis=0)

    print(f"Best Train Loss: {best_train_loss}, Best Val Loss: {best_val_loss}")
    print(f"Test Loss: {loss_test}")

    results_dict = {
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "loss_test": loss_test,
        "train_time": end_time - start_time,
        "mean_fct_intercept_class_0": mean_fct_intercept[0],
        "mean_fct_intercept_class_1": mean_fct_intercept[1],
        "mean_fct_intercept_class_2": mean_fct_intercept[2],
        "mean_fct_intercept_class_3": mean_fct_intercept[3],
        "mean_learnt_fct_intercept_class_0": mean_learnt_fct_intercept[0],
        "mean_learnt_fct_intercept_class_1": mean_learnt_fct_intercept[1],
        "mean_learnt_fct_intercept_class_2": mean_learnt_fct_intercept[2],
        "mean_learnt_fct_intercept_class_3": mean_learnt_fct_intercept[3],
        "mean_fct_intercept_test_class_0": mean_fct_intercept_test[0],
        "mean_fct_intercept_test_class_1": mean_fct_intercept_test[1],
        "mean_fct_intercept_test_class_2": mean_fct_intercept_test[2],
        "mean_fct_intercept_test_class_3": mean_fct_intercept_test[3],
        "mean_learnt_fct_intercept_test_class_0": mean_learnt_fct_intercept_test[0],
        "mean_learnt_fct_intercept_test_class_1": mean_learnt_fct_intercept_test[1],
        "mean_learnt_fct_intercept_test_class_2": mean_learnt_fct_intercept_test[2],
        "mean_learnt_fct_intercept_test_class_3": mean_learnt_fct_intercept_test[3],
    }

    for i in range(n_alternatives):
        results_dict[f"L1_distance_train_class_{i}"] = l1_distance_train[i]
        results_dict[f"L1_distance_test_class_{i}"] = l1_distance_test[i]

    if args.save_model:
        # save the results
        pd.DataFrame(results_dict, index=[0]).to_csv(
            args.outpath
            + f"results_dict_fi{args.functional_intercept}_fp{args.functional_params}.csv"
        )

        model.save_model(save_path)


def hyperparameter_search(model: str = "RUMBoost") -> None:
    """
    Perform hyperparameter search for the models.
    This function is not implemented yet.

    Parameters
    ----------
    model : str
        The model to train. Can be "RUMBoost" or "TasteNet".
    """
    # reset all seeds at the beginning of each experiment for reproducibility
    set_all_seeds(0)

    # load data
    data_train, _ = create_dataset()

    # create synthetic utility values and choices
    data_train = add_simulated_choices(
        data_train,
        with_noise=True,
    )

    X, y = data_train[features], data_train["choice"]

    # ensuring a split that does not have individuals in both train and validation sets
    train_indices, val_indices = np.arange(int(X.shape[0] * 0.8)), np.arange(
        int(X.shape[0] * 0.8), X.shape[0]
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

    func_int = True
    func_params = False

    objective = partial(
        objective,
        model=model,
        func_int=func_int,
        func_params=func_params,
    )

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0)
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

    dataset = "synthetic"
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


def plot_ind_spec_constant(
    save_fig: bool = True,
    functional_params: bool = False,
    functional_intercept: bool = True,
) -> None:
    """
    Plot the individual-specific constant for the models.
    The model needs to be trained with functional parameters or functional intercept.

    Parameters
    ----------
    save_fig : bool
        Whether to save the figure or not.
    functional_params : bool
        If the model is trained with functional parameters.
    functional_intercept : bool
        If the model is trained with functional intercept.
    """
    # load data
    df_train, df_test = create_dataset()

    # Plot the features
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        "legend.borderaxespad": 0.4,
        "legend.borderpad": 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    # sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            # "font.sans-serif": "Computer Modern Roman",
        }
    )

    num_plots = n_alternatives - 1

    true_fct_intercepts_train = create_functional_effects(
        df_train.values, n_alternatives, len(socio_demo_chars)
    )
    true_fct_intercepts_test = create_functional_effects(
        df_test.values, n_alternatives, len(socio_demo_chars)
    )

    for t in ["train", "test"]:
        if t == "train":
            df = df_train
            true_fct_intercepts = true_fct_intercepts_train
        else:
            df = df_test
            true_fct_intercepts = true_fct_intercepts_test
        # Load the models
        for model in all_models.keys():
            if model == "RUMBoost":
                model_path = f"results/synthetic/{model}/model_fi{functional_intercept}_fp{functional_params}.json"
                rumboost = all_models[model]()
                rumboost.load_model(model_path)
                y_rumboost = gather_functional_intercepts(
                    df, rumboost, socio_demo_chars, n_alternatives, alt_normalised=3
                )
            elif model == "TasteNet":
                model_path = f"results/synthetic/{model}/model_fi{functional_intercept}_fp{functional_params}.pth"
                tastenet = all_models[model]()
                tastenet.load_model(path=model_path)
                y_tastenet = gather_functional_intercepts(
                    df, tastenet, socio_demo_chars, n_alternatives, alt_normalised=3
                )
            elif model == "MixedEffect":
                model_path = f"results/synthetic/{model}/model_fi{functional_intercept}_fp{functional_params}.yaml"
                mixedeffect = all_models[model]()
                mixedeffect.load_model(path=model_path, alt_spec_features=alt_spec_features, socio_demo_chars=socio_demo_chars, num_classes=n_alternatives)
                df["ID"] = np.repeat(
                    np.arange(int(df.shape[0] / panel_factor)), panel_factor
                )
                mixedeffect.build_dataloader(df, pd.Series(np.random.randint(low=0, high=4, size=df.shape[0])), None, None)

                y_mixedeffect = gather_functional_intercepts(
                    df,
                    mixedeffect,
                    socio_demo_chars,
                    n_alternatives,
                    alt_normalised=3,
                    on_train_set=(t == "train"),
                )

        colors = ["#264653", "#2a9d8f", "#0073a1", "#7cd2bf"]

        for j in range(num_plots):
            fig, axes = plt.subplots(1, 4, figsize=(10, 2), dpi=300)
            # fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
            if "RUMBoost" in all_models:
                min_val = y_rumboost[:, j].min()
                max_val = y_rumboost[:, j].max()
            if "TasteNet" in all_models:
                min_val = min(min_val, y_tastenet[:, j].min())
                max_val = max(max_val, y_tastenet[:, j].max())
            if "MixedEffect" in all_models:
                min_val = min(min_val, y_mixedeffect[:, j].min())
                max_val = max(max_val, y_mixedeffect[:, j].max())

            min_val = min(min_val, true_fct_intercepts[:, j].min())
            max_val = max(max_val, true_fct_intercepts[:, j].max())

            bin_edges = np.linspace(min_val, max_val, 50)

            max_count = np.histogram(true_fct_intercepts[:, j], bins=bin_edges)[0].max()
            for model in all_models.keys():
                if model == "RUMBoost":
                    counts, _ = np.histogram(y_rumboost[:, j], bins=bin_edges)
                elif model == "TasteNet":
                    counts, _ = np.histogram(y_tastenet[:, j], bins=bin_edges)
                elif model == "MixedEffect":
                    counts, _ = max_count, None
                max_count = max(max_count, counts.max())

            # sns.histplot(
            #     true_fct_intercepts[:, j],
            #     ax=axes,
            #     bins=bin_edges,
            #     color="black",
            #     label="True functional intercept",
            #     alpha=1
            # )
            # sns.histplot(
            #     y_rumboost[:, j],
            #     ax=axes,
            #     bins=bin_edges,
            #     color=colors[0],
            #     label=f"GBDT (MAE: {l1_distance(true_fct_intercepts[:,j], y_rumboost[:, j])/y_rumboost.shape[0]:.2f})",
            #     alpha=0.7
            # )
            # sns.histplot(
            #     y_tastenet[:, j],
            #     ax=axes,
            #     bins=bin_edges,
            #     color=colors[1],
            #     label=f"DNN (MAE: {l1_distance(true_fct_intercepts[:,j], y_tastenet[:, j])/y_tastenet.shape[0]:.2f})",
            #     alpha=0.7
            # )
            # axes.set_ylabel("Count")
            # axes.legend()

            # xlim = (
            #     min_val - (max_val - min_val) * 0.01,
            #     max_val + (max_val - min_val) * 0.01,
            # )
            # ylim = (0, max_count * 1.1)

            # plt.setp(axes, xlim=xlim, ylim=ylim)

            for i, (model, ax) in enumerate(
                zip(
                    ["True functional intercept"] + list(all_models.keys()),
                    axes.flatten(),
                )
            ):
                if model == "True functional intercept":
                    sns.histplot(
                        true_fct_intercepts[:, j],
                        ax=ax,
                        bins=bin_edges,
                        color="black",
                        label="True functional intercept",
                    )
                elif model == "RUMBoost":
                    sns.histplot(
                        y_rumboost[:, j],
                        ax=ax,
                        bins=bin_edges,
                        color=colors[i - 1],
                        label=f"GBDT (MAE: {l1_distance(true_fct_intercepts[:,j], y_rumboost[:, j])/y_rumboost.shape[0]:.2f})",
                    )
                elif model == "TasteNet":
                    sns.histplot(
                        y_tastenet[:, j],
                        ax=ax,
                        bins=bin_edges,
                        color=colors[i - 1],
                        label=f"DNN (MAE: {l1_distance(true_fct_intercepts[:,j], y_tastenet[:, j])/y_tastenet.shape[0]:.2f})",
                    )
                elif model == "MixedEffect":
                    sns.histplot(
                        y_mixedeffect[:, j],
                        ax=ax,
                        bins=bin_edges,
                        color="#B35733",
                        label=f"MixedLogit (MAE: {l1_distance(true_fct_intercepts[:,j], y_mixedeffect[:, j])/y_mixedeffect.shape[0]:.2f})",
                    )

                if i == 0:
                    ax.set_ylabel("Count")
                else:
                    ax.set_ylabel("")

                xlim = (
                    min_val - (max_val - min_val) * 0.01,
                    max_val + (max_val - min_val) * 0.01,
                )
                ylim = (0, max_count * 1.1)

                plt.setp(axes, xlim=xlim, ylim=ylim)

                ax.legend()

            if save_fig:
                save_path = (
                    f"results/synthetic/figures/intercept_{j}_fiTrue_fpFalse_{t}.png"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_alt_spec_features(
    alt_spec_features_list: list[str],
    save_fig: bool = True,
) -> None:
    """
    Plot the alternative-specific features for the models, if trained without functional parameters.

    Parameters
    ----------
    alt_spec_features_list : list[str]
        List of alternative-specific features. They must be in the same order as for the training.
    save_fig : bool
        Whether to save the figure or not.
    """
    num_classes = n_alternatives
    # Load the models
    for model in all_models.keys():
        if model == "RUMBoost":
            model_path_fi = f"results/synthetic/{model}/model_fiTrue_fpFalse.json"
            rumboost_fi = all_models[model]()
            rumboost_fi.load_model(model_path_fi)
            rumboost_params_fi = rumboost_fi.model.boosters[:-num_classes]
        elif model == "TasteNet":
            model_path_fi = f"results/synthetic/{model}/model_fiTrue_fpFalse.pth"
            tastenet_fi = all_models[model]()
            tastenet_fi.load_model(path=model_path_fi)
            tastenet_params_fi = np.array([])
            for param in tastenet_fi.model.util_module.mnl.mnl.parameters():
                tastenet_params_fi = np.concatenate(
                    [tastenet_params_fi, param.detach().cpu().numpy()[0]]
                )
        elif model == "MixedEffect":
            model_path_fi = f"results/synthetic/{model}/model_fiTrue_fpFalse.yaml"
            mixedeffect_fi = all_models[model]()
            mixedeffect_fi.load_model(path=model_path_fi)
            mixedeffect_params_fi = mixedeffect_fi.params

    # Plot the features
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        "legend.borderaxespad": 0.4,
        "legend.borderpad": 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    # sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            # "font.sans-serif": "Computer Modern Roman",
        }
    )

    colors = ["#264653", "#2a9d8f", "#0073a1", "#7cd2bf"]

    for i, as_feat in enumerate(alt_spec_features_list):

        x = np.linspace(0, 1, 10000)
        dummy_array = np.zeros((10000, len(alt_spec_features_list)))
        dummy_array[:, i] = x
        y_rumboost = rumboost_params_fi[i].predict(dummy_array[:, i].reshape(-1, 1))
        y_tastenet = tastenet_params_fi[i] * x
        y_mixedeffect = mixedeffect_params_fi[mixedeffect_params_fi["Name"] == f"beta_{as_feat}_alt{i}"]["Value"].values * x

        y_rumboost = [y - y_rumboost[0] for y in y_rumboost]

        y_true = -x

        # Plot the features
        plt.figure(figsize=(2.62, 1.97), dpi=300)

        plt.plot(x, y_rumboost, label="FI-RUMBoost", color=colors[2], linewidth=0.8)

        plt.plot(x, y_tastenet, label="FI-DNN", color=colors[3], linewidth=0.8)

        plt.plot(
            x, y_mixedeffect, label="MixedLogit", color="#B35733", linewidth=0.8
        )

        plt.plot(x, y_true, label="True function", color="black", linewidth=0.8)

        # plt.xlabel(feature_names[as_feat])
        plt.ylabel("Utility")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            save_path = f"results/synthetic/figures/{as_feat}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    for model in all_models.keys():
        # run hyperparameter search
        # hyperparameter_search(model=model)

        # load the optimal hyperparameters for the model
        args = parse_cmdline_args()
        if model != "MixedEffect":
            try:
                opt_hyperparams_path = (
                    f"results/synthetic/{model}/best_params_fiTrue_fpFalse.pkl"
                )
                with open(opt_hyperparams_path, "rb") as f:
                    optimal_hyperparams = pickle.load(f)
                    if "layer_sizes" in optimal_hyperparams:
                        optimal_hyperparams["layer_sizes"] = [
                            int(size)
                            for size in optimal_hyperparams["layer_sizes"].split(",")
                        ]
                    if "learning_rate" not in optimal_hyperparams:
                        optimal_hyperparams["learning_rate"] = 1
                    args.__dict__.update(optimal_hyperparams)
            except FileNotFoundError:
                print(
                    f"Optimal hyperparameters not found for {model}. Running hyperparameter search."
                )
                hyperparameter_search(model=model)
                opt_hyperparams_path = (
                    f"results/synthetic/{model}/best_params_fiTrue_fpFalse.pkl"
                )
                with open(opt_hyperparams_path, "rb") as f:
                    optimal_hyperparams = pickle.load(f)
                    if "layer_sizes" in optimal_hyperparams:
                        optimal_hyperparams["layer_sizes"] = [
                            int(size)
                            for size in optimal_hyperparams["layer_sizes"].split(",")
                        ]
                    if "learning_rate" not in optimal_hyperparams:
                        optimal_hyperparams["learning_rate"] = 1
                    args.__dict__.update(optimal_hyperparams)
        args.functional_intercept = True
        args.functional_params = False
        args.save_model = True
        args.model = model
        args.dataset = "synthetic"
        if model == "RUMBoost":
            args.num_iterations = int(args.best_iteration)
        elif model == "TasteNet":
            args.num_epochs = int(args.best_iteration)
        args.early_stopping_rounds = None
        run_experiment(args)

    plot_ind_spec_constant()
    plot_alt_spec_features(["f4", "f5", "f6", "f7"])
