# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import torch
from scipy.special import softmax
from rumboost.metrics import cross_entropy
from rumboost.datasets import load_preprocess_LPMC
import time
import os
import pickle

from models_wrapper import RUMBoost, TasteNet
from parser import parse_cmdline_args
from utils import split_dataset

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


def create_discontinuity(x, x_disc, jump):
    return np.where(x < x_disc, x, 0.5 * (x - x_disc) + x_disc + jump)


# Define the utility function
def utility_function_LPMC(data, with_noise=False):
    # Extract the parameters
    V = np.zeros((data.shape[0], n_alternatives))

    V[:, 0] = (
        create_functional_intercept(data, ["age", "female"]) + -1 * data["dur_walking"]
    )
    V[:, 1] = (
        create_functional_intercept(data, ["age", "car_ownership"])
        + -1 * data["dur_cycling"]
    )
    V[:, 2] = (
        create_functional_intercept(data, ["age", "driving_license"])
        + -1 * data["dur_pt_rail"]
    )
    V[:, 3] = (
        create_functional_intercept(
            data, ["female", "car_ownership", "driving_license"]
        )
        + -1 * data["dur_driving"]
    )

    if with_noise:
        noise = generate_noise(0, 1, (data.shape[0], n_alternatives))
        V += noise

    return V


def generate_noise(mean, sd, n):
    return np.random.gumbel(loc=mean, scale=sd, size=n)


def compute_prob(V):

    return softmax(V, axis=1)


def generate_labels(probs):
    labels = [
        np.random.choice(range(n_alternatives), p=probs[i])
        for i in range(probs.shape[0])
    ]
    return np.array(labels)

def group_functional_intercepts(data_train: pd.DataFrame, data_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
            fct_intercept_0,
            fct_intercept_1,
            fct_intercept_2,
            fct_intercept_3,
        ]
    ).T
    fct_intercepts_test = np.array(
        [
            fct_intercept_0_test,
            fct_intercept_1_test,
            fct_intercept_2_test,
            fct_intercept_3_test,
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

def gather_functional_intercepts(data: pd.DataFrame, features_name: list, model) -> np.ndarray:
    """
    Gather the learnt functional intercepts for the given model.

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    features_name: list
        Features used in the functional intercept

    Returns
    -------
    functional_intercepts: np.ndarray
        The functional intercepts for the given features.
    """
    if isinstance(model, RUMBoost):
        rumboost_predictor = model.model.boosters[-n_alternatives:]
        fct_intercept = np.zeros((data.shape[0], n_alternatives))
        for i, predictor in enumerate(rumboost_predictor):
            fct_intercept[:,i] = predictor.predict(data[socio_demo_chars])
        
    else:
        tastenet_predictor = model.model.params_module
        # already computing the functional values as they are outputted all at once
        sdc_tensor = (
            torch.from_numpy(data[socio_demo_chars].values)
            .to(torch.device("cuda"))
            .to(torch.float32)
        )
        fct_intercept = tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()

    return np.minimum(fct_intercept, 0)

def l1_distance(true_fct_intercept, learnt_fct_intercept):
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

def run_experiment(args):
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
    V_train_noisy = utility_function_LPMC(data_train, with_noise=False)
    V_test_noisy = utility_function_LPMC(data_test, with_noise=False)
    simulated_probs = compute_prob(V_train_noisy)
    simulated_choice = generate_labels(simulated_probs)
    simulated_probs_test = compute_prob(V_test_noisy)
    simulated_choice_test = generate_labels(simulated_probs_test)
    data_train["choice"] = simulated_choice
    data_test["choice"] = simulated_choice_test

    # split dataset by household id
    X_train, y_train, X_val, y_val = split_dataset(
        data_train,
        "choice",
        features,
        train_size=0.8,
        groups=data_train["household_id"],
        random_state=1,
    )
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

    #fit model
    start_time = time.time()
    best_train_loss, best_val_loss = model.fit()
    end_time = time.time()

    # predict on the test set
    preds, _, _ = model.predict(X_test)
    loss_test = cross_entropy(preds, y_test)

    # get learnt functional intercepts
    learnt_fct_intercepts = gather_functional_intercepts(
        data_train, socio_demo_chars, model
    )
    learnt_fct_intercepts_test = gather_functional_intercepts(
        data_test, socio_demo_chars, model
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

def plot_ind_spec_constant(
    save_fig: bool = True,
    functional_params: bool = False,
    functional_intercept: bool = True,
):
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
                y_tastenet = tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()
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

            min_val = min(min_val, true_fct_intercepts[:,j].min())
            max_val = max(max_val, true_fct_intercepts[:,j].max())

            bin_edges = np.linspace(min_val, max_val, 50)

            max_count = np.histogram(
                true_fct_intercepts[:, j], bins=bin_edges
            )[0].max()
            for model in all_models.keys():
                if model == "RUMBoost":
                    counts, _ = np.histogram(y_rumboost, bins=bin_edges)
                elif model == "TasteNet":
                    counts, _ = np.histogram(y_tastenet[:, j], bins=bin_edges)
                max_count = max(max_count, counts.max())

            sns.histplot(
                true_fct_intercepts[:,j],
                ax=axes,
                bins=bin_edges,
                color="black",
                label="True Functional Intercept",
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
                save_path = f"results/synthetic/figures/intercept_{j}_fiTrue_fpFalse_{t}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == "__main__":

    args = parse_cmdline_args()
    for model in all_models.keys():
        args.model = model
        args.dataset = "synthetic"
        # load the optimal hyperparameters for the model
        try:
            opt_hyperparams_path = f"results/SwissMetro/{model}/best_params_fiTrue_fpFalse.pkl"
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
                args.num_iterations = 3000
                args.early_stopping_rounds = 100
                args.num_epochs = 200
                args.patience = 10
        except FileNotFoundError:
            print(
                f"Optimal hyperparameters not found for {args.model}. Using default hyperparameters."
            )
            optimal_hyperparams = None
        args.functional_intercept = True
        args.functional_params = False
        args.save_model = True
        print(args.model)
        run_experiment(args)

    plot_ind_spec_constant()