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

from functools import partial
from scipy.special import softmax
from rumboost.metrics import cross_entropy
from rumboost.datasets import load_preprocess_LPMC

from models_wrapper import RUMBoost, TasteNet
from parser import parse_cmdline_args

np.random.seed(1)

n_alternatives = 4
features = [
    "age",
    "female",
    "car_ownership",
    "driving_license",
    "dur_walking",
    "dur_cycling",
    "dur_pt_rail",
    "dur_driving",
]
alt_spec_features = {
    0: ["dur_walking"],
    1: ["dur_cycling"],
    2: ["dur_pt_rail"],
    3: ["dur_driving"],
}
socio_demo_chars = [
    "age",
    "female",
    "car_ownership",
    "driving_license",
]
all_models = {
    "RUMBoost": RUMBoost,
    "TasteNet": TasteNet,
}


# Define the utility function
def utility_function_LPMC(data: pd.DataFrame, with_noise: bool = False) -> np.ndarray:
    """
    Create the utility function for the LPMC dataset.

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

    fct_int_0 = create_functional_intercept(data, ["age", "female"])
    fct_int_1 = create_functional_intercept(data, ["age", "car_ownership"])
    fct_int_2 = create_functional_intercept(data, ["age", "driving_license"])
    fct_int_3 = create_functional_intercept(
        data,
        ["female", "car_ownership", "driving_license"],
    )

    V[:, 0] = fct_int_0 / fct_int_0.max() + -1 * data["dur_walking"]
    V[:, 1] = fct_int_1 / fct_int_1.max() + -1 * data["dur_cycling"]
    V[:, 2] = fct_int_2 / fct_int_2.max() + -1 * data["dur_pt_rail"]
    V[:, 3] = fct_int_3 / fct_int_3.max() + -1 * data["dur_driving"]

    if with_noise:
        noise = generate_noise(0, 1, (data.shape[0], n_alternatives))
        V += noise

    return V


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


def group_functional_intercepts(
    data_train: pd.DataFrame, data_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the synthetic functional intercepts for the LPMC dataset.
    Parameters
    ----------
    data_train: pd.DataFrame
        Data used for the synthetic experiment
    data_test: pd.DataFrame
        Data used for the synthetic experiment

    Returns
    -------
    fct_intercepts: Tuple[np.ndarray, np.ndarray]
        The functional intercepts for the training and test sets.
    """
    fct_intercept_0 = create_functional_intercept(data_train, ["age", "female"])
    fct_intercept_0_test = create_functional_intercept(data_test, ["age", "female"])
    fct_intercept_1 = create_functional_intercept(data_train, ["age", "car_ownership"])
    fct_intercept_1_test = create_functional_intercept(
        data_test, ["age", "car_ownership"]
    )
    fct_intercept_2 = create_functional_intercept(
        data_train, ["age", "driving_license"]
    )
    fct_intercept_2_test = create_functional_intercept(
        data_test, ["age", "driving_license"]
    )
    fct_intercept_3 = create_functional_intercept(
        data_train, ["female", "car_ownership", "driving_license"]
    )
    fct_intercept_3_test = create_functional_intercept(
        data_test, ["female", "car_ownership", "driving_license"]
    )
    fct_intercepts_train = np.array(
        [
            fct_intercept_0 / fct_intercept_0.max(),
            fct_intercept_1 / fct_intercept_1.max(),
            fct_intercept_2 / fct_intercept_2.max(),
            fct_intercept_3 / fct_intercept_3.max(),
        ]
    ).T
    fct_intercepts_test = np.array(
        [
            fct_intercept_0_test / fct_intercept_0_test.max(),
            fct_intercept_1_test / fct_intercept_1_test.max(),
            fct_intercept_2_test / fct_intercept_2_test.max(),
            fct_intercept_3_test / fct_intercept_3_test.max(),
        ]
    ).T
    return fct_intercepts_train, fct_intercepts_test


def create_functional_intercept(data: pd.DataFrame, features_name: list) -> np.ndarray:
    """
    Create the synthetic functional intercepts

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    features_name: list
        Features used in the functional intercept

    Returns
    -------
    functional_intercept: np.ndarray
        The functional intercepts
    """
    data_arr = data[features_name].values
    functional_intercept = data_arr.prod(axis=1)
    return functional_intercept


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
    V = utility_function_LPMC(data, with_noise=with_noise)
    probs = compute_prob(V)
    data["choice"] = generate_labels(probs)
    return data


def gather_functional_intercepts(
    data: pd.DataFrame,
    model: RUMBoost | TasteNet,
    socio_demo_characts: list[str] = socio_demo_chars,
    n_classes: int = n_alternatives,
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

    Returns
    -------
    functional_intercepts: np.ndarray
        The functional intercepts for the given features.
    """
    if isinstance(model, RUMBoost):
        rumboost_predictor = model.model.boosters[-n_classes:]
        fct_intercept = np.zeros((data.shape[0], n_classes))
        for i, predictor in enumerate(rumboost_predictor):
            fct_intercept[:, i] = predictor.predict(data[socio_demo_characts], raw_score=True)
    else:
        tastenet_predictor = model.model.params_module
        # already computing the functional values as they are outputted all at once
        sdc_tensor = (
            torch.from_numpy(data[socio_demo_characts].values)
            .to(torch.device("cuda"))
            .to(torch.float32)
        )
        fct_intercept = tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()

    return fct_intercept


def l1_distance(true_fct_intercept: np.ndarray, learnt_fct_intercept: np.ndarray) -> float:
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
    return np.sum(np.abs(true_fct_intercept - learnt_fct_intercept))


def run_experiment(args: argparse.Namespace) -> None:
    """
    Run the synthetic experiment with the given arguments.

    Parameters
    ----------
    args: argparse.Namespace
        The command line arguments parsed by the parser.
    """
    args.outpath = f"results/synthetic/{args.model}/"
    os.makedirs(args.outpath, exist_ok=True)

    # load data
    data_train, data_test, _ = load_preprocess_LPMC(path="../data/LPMC/")

    # create synthetic utility values and choices
    data_train = add_simulated_choices(
        data_train,
        with_noise=False,
    )
    data_test = add_simulated_choices(
        data_test,
        with_noise=False,
    )

    X_train, y_train = data_train[features], data_train["choice"]
    X_val, y_val = None, None
    X_test, y_test = data_test[features], data_test["choice"]

    # create synthetic "ground truth" functional intercepts
    fct_intercept, fct_intercept_test = group_functional_intercepts(
        data_train,
        data_test,
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
    best_train_loss, best_val_loss = model.fit()
    end_time = time.time()

    # predict on the test set
    preds, _, _ = model.predict(X_test)
    loss_test = cross_entropy(preds, y_test)

    # get learnt functional intercepts
    learnt_fct_intercepts = gather_functional_intercepts(
        data_train, model, socio_demo_chars, n_alternatives
    )
    learnt_fct_intercepts_test = gather_functional_intercepts(
        data_test, model, socio_demo_chars, n_alternatives
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

    print(f"Best Train Loss: {best_train_loss}, Best Val Loss: {best_val_loss}")
    print(f"Test Loss: {loss_test}")

    results_dict = {
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "loss_test": loss_test,
        "l1_distance_train": l1_distance_train,
        "l1_distance_test": l1_distance_test,
        "train_time": end_time - start_time,
    }

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

    def objective(trial: optuna.Trial, model: str, func_int: bool, func_params: bool) -> float:
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

        # load the data
        data, _, folds = load_preprocess_LPMC(path="../data/LPMC/")

        data = add_simulated_choices(
            data,
            features_name=features,
            with_noise=False,
        )

        X, y = data[features], data["choice"]

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

        avg_val_loss = 0.0
        avg_best_iter = 0.0
        k = 1
        for i, (train_idx, val_idx) in enumerate(folds):
            if i > 0:
                continue  # can only do 1 fold because of computational time
            # split data
            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

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

    func_int = True
    func_params = False

    objective = partial(
        objective,
        model=model,
        func_int=func_int,
        func_params=func_params,
    )

    study = optuna.create_study(direction="minimize")

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
    df_train, df_test, _ = load_preprocess_LPMC(path="../data/LPMC/")

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

    num_plots = n_alternatives

    true_fct_intercepts_train, true_fct_intercepts_test = group_functional_intercepts(
        df_train,
        df_test,
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
                rumboost_predictor = rumboost.model.boosters[-num_plots:]

            elif model == "TasteNet":
                model_path = f"results/synthetic/{model}/model_fi{functional_intercept}_fp{functional_params}.pth"
                tastenet = all_models[model]()
                tastenet.load_model(path=model_path)
                tastenet_predictor = tastenet.model.params_module
                # already computing the functional values as they are outputted all at once
                sdc_tensor = (
                    torch.from_numpy(df[socio_demo_chars].values)
                    .to(torch.device("cuda"))
                    .to(torch.float32)
                )
                y_tastenet = (
                    tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()
                )
                if not functional_params:
                    y_tastenet = y_tastenet.reshape(-1, n_alternatives)

        colors = ["#264653", "#2a9d8f", "#0073a1", "#7cd2bf"]

        for j in range(num_plots):
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
            if "RUMBoost" in all_models:
                y_rumboost = rumboost_predictor[j].predict(df[socio_demo_chars])
                min_val = y_rumboost.min()
                max_val = y_rumboost.max()
            if "TasteNet" in all_models:
                min_val = min(min_val, y_tastenet[:, j].min())
                max_val = max(max_val, y_tastenet[:, j].max())

            min_val = min(min_val, true_fct_intercepts[:, j].min())
            max_val = max(max_val, true_fct_intercepts[:, j].max())

            bin_edges = np.linspace(min_val, max_val, 50)

            max_count = np.histogram(true_fct_intercepts[:, j], bins=bin_edges)[0].max()
            for model in all_models.keys():
                if model == "RUMBoost":
                    counts, _ = np.histogram(y_rumboost, bins=bin_edges)
                elif model == "TasteNet":
                    counts, _ = np.histogram(y_tastenet[:, j], bins=bin_edges)
                max_count = max(max_count, counts.max())

            sns.histplot(
                true_fct_intercepts[:, j],
                ax=axes,
                bins=bin_edges,
                color="black",
                label="True functional intercept",
            )

            for i, model in enumerate(all_models.keys()):
                ax = axes
                if model == "RUMBoost":
                    sns.histplot(
                        y_rumboost,
                        ax=ax,
                        bins=bin_edges,
                        color=colors[i],
                        label=f"{model} (L1: {l1_distance(true_fct_intercepts[:,j], y_rumboost):.2f})",
                    )
                elif model == "TasteNet":
                    sns.histplot(
                        y_tastenet[:, j],
                        ax=ax,
                        bins=bin_edges,
                        color=colors[i],
                        label=f"{model} (L1: {l1_distance(true_fct_intercepts[:,j], y_tastenet[:, j]):.2f})",
                    )

            title = model
            fig_title = "Intercept"
            ax.set_title(title)
            ax.set_ylabel("Count")
            plt.title(fig_title, fontsize=8)

            xlim = (
                min_val - (max_val - min_val) * 0.01,
                max_val + (max_val - min_val) * 0.01,
            )
            ylim = (0, max_count * 1.1)

            plt.setp(axes, xlim=xlim, ylim=ylim)
            plt.legend()

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

        x = np.linspace(0, 3, 10000)
        dummy_array = np.zeros((10000, len(alt_spec_features_list)))
        dummy_array[:, i] = x
        y_rumboost = rumboost_params_fi[i].predict(dummy_array[:, i].reshape(-1, 1))
        y_tastenet = tastenet_params_fi[i] * x

        y_rumboost = [y - y_rumboost[0] for y in y_rumboost]

        y_true = -x

        # Plot the features
        plt.figure(figsize=(2.62, 1.97), dpi=300)

        plt.plot(x, y_rumboost, label="RUMBoost", color=colors[2], linewidth=0.8)

        plt.plot(x, y_tastenet, label="TasteNet", color=colors[3], linewidth=0.8)

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
                f"Optimal hyperparameters not found for {model}. Using default hyperparameters."
            )
            optimal_hyperparams = None
        args.functional_intercept = True
        args.functional_params = False
        args.save_model = True
        args.model = model
        args.dataset = "synthetic"
        if model == "RUMBoost":
            args.device = "cpu"
            args.num_iterations = int(args.best_iteration)
            args.num_iterations = 1000
        else:
            args.device = "cuda"
            args.num_epochs = int(args.best_iteration)
        args.early_stopping_rounds = None
        print(args.lambda_l1)
        print(args.learning_rate)
        print(args.num_leaves)
        print(args.act_func)
        run_experiment(args)

    plot_ind_spec_constant()
    plot_alt_spec_features(["dur_walking", "dur_cycling", "dur_pt_rail", "dur_driving"])
