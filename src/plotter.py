import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os

from models_wrapper import RUMBoost, ResLogit, TasteNet
from constants import alt_spec_features, PATH_TO_DATA
from typing import List, Dict


all_models = {
    "RUMBoost": RUMBoost,
    "ResLogit": ResLogit,
    "TasteNet": TasteNet,
}


def plot_alt_spec_features(
    alt_spec_features: List = alt_spec_features,
    all_models: Dict = all_models,
    path_to_data: str = PATH_TO_DATA,
    save_fig: bool = False,
):
    """
    Plot the alternative-specific features for the models.

    Parameters
    ----------
    alt_spec_features : List
        List of alternative-specific features. They must be in the same order as for the training.
    all_models : Dict
        Dictionary of all models.
    path_to_data : str
        Path to the data folder.
    save_fig : bool
        Whether to save the figure or not.
    """
    # Load the models
    for model in all_models.keys():
        if model == "RUMBoost":
            model_path = f"results/{model}/model.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            rumboost_params = rumboost.model.boosters[:-1]
        elif model == "ResLogit":
            model_path = f"results/{model}/model.pth"
            reslogit = all_models[model]()
            reslogit.load_model(path=model_path)
            reslogit_params = [
                params.detach().cpu().numpy()
                for params in reslogit.model.mnl_as.parameters()
            ][:-1][0]
        elif model == "TasteNet":
            model_path = f"results/{model}/model.pth"
            tastenet = all_models[model]()
            tastenet.load_model(path=model_path)
            tastenet_params = [
                params.detach().cpu().numpy()
                for params in tastenet.model.util_module.mnl.parameters()
            ][:-1][0]

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

    df = pd.read_csv(path_to_data)

    colors = ["#264653", "#2a9d8f", "#f4a261"]

    for i, as_feat in enumerate(alt_spec_features):

        x = np.linspace(0, df[as_feat].max(), 10000)
        dummy_array = np.zeros((10000, len(alt_spec_features)))
        dummy_array[:, i] = x
        y_rumboost = rumboost_params[0].predict(dummy_array)
        y_reslogit = reslogit_params[:, i] * x
        y_tastenet = tastenet_params[:, i] * x

        y_rumboost = [y - y_rumboost[0] for y in y_rumboost]

        # Plot the features
        plt.figure(figsize=(2.62, 1.97), dpi=300)

        plt.plot(x, y_rumboost, label="RUMBoost", color=colors[0], linewidth=0.8)
        plt.plot(x, y_reslogit, label="ResLogit", color=colors[1], linewidth=0.8)
        plt.plot(x, y_tastenet, label="TasteNet", color=colors[2], linewidth=0.8)
        plt.xlabel(as_feat)
        plt.ylabel("Utility")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            save_path = f"results/figures/{as_feat}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def plot_ind_spec_constant(
    alt_spec_features: List = alt_spec_features,
    all_models: Dict = all_models,
    path_to_data: str = PATH_TO_DATA,
    save_fig: bool = False,
    feature_to_highlight: str = None,
):
    """
    Plot the individual-specific constant for the models.

    Parameters
    ----------
    alt_spec_features : List
        List of alternative-specific features. They must be in the same order as for the training.
    all_models : Dict
        Dictionary of all models.
    path_to_data : str
        Path to the data folder.
    save_fig : bool
        Whether to save the figure or not.
    feature_to_highlight : str
        Feature to highlight in the plot. If None, no feature is highlighted.
    """
    # Load the models
    for model in all_models.keys():
        if model == "RUMBoost":
            model_path = f"results/{model}/model.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            rumboost_predictor = rumboost.model.boosters[-1]
        elif model == "ResLogit":
            model_path = f"results/{model}/model.pth"
            reslogit = all_models[model]()
            reslogit.load_model(path=model_path)
            reslogit_predictor = reslogit.model
        elif model == "TasteNet":
            model_path = f"results/{model}/model.pth"
            tastenet = all_models[model]()
            tastenet.load_model(path=model_path)
            tastenet_predictor = tastenet.model.params_module

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

    df = pd.read_csv(path_to_data)

    socio_demo_chars = [
        col
        for col in df.columns
        if col not in alt_spec_features
        and col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]

    colors = ["#264653", "#2a9d8f", "#f4a261"]

    fig, axes = plt.subplots(1, 3, figsize=(8, 6), dpi=300)

    for i, (model, ax) in enumerate(zip(all_models.keys(), axes.flatten())):
        if feature_to_highlight:
            df[feature_to_highlight] = df[feature_to_highlight].astype("category")
            df[feature_to_highlight] = df[feature_to_highlight].cat.codes
            hue = df[feature_to_highlight]
        else:
            hue = None
        if model == "RUMBoost":
            y_rumboost = rumboost_predictor.predict(df[socio_demo_chars])
            sns.histplot(
                y_rumboost,
                ax=ax,
                bins=50,
                color=colors[i],
                hue=hue,
                # multiple="stack",
            )
        elif model == "ResLogit":
            sdc_tensor = torch.from_numpy(df[socio_demo_chars].values).to(
                torch.device("cuda")
            ).to(torch.float32)
            y_reslogit = (
                reslogit_predictor.resnet_layer(reslogit_predictor.mnl_sd(sdc_tensor))
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            sns.histplot(
                y_reslogit,
                ax=ax,
                bins=50,
                color=colors[i],
                hue=hue,
                # multiple="stack",
            )
        elif model == "TasteNet":
            sdc_tensor = torch.from_numpy(df[socio_demo_chars].values).to(
                torch.device("cuda")
            ).to(torch.float32)
            y_tastenet = tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()
            sns.histplot(
                y_tastenet,
                ax=ax,
                bins=50,
                color=colors[i],
                hue=hue,
                # multiple="stack",
            )
        ax.set_title(model)
        ax.set_xlabel("Individual-specific constant")
        ax.set_ylabel("Count")

    # # Defining custom 'xlim' and 'ylim' values.
    # xlim = (-3.5, 3.5)
    # ylim = (0, 5250)

    # # Setting the values for all axes.
    # plt.setp(axes, xlim=xlim, ylim=ylim)

    if save_fig:
        if not feature_to_highlight:
            feature_to_highlight = "all"
        save_path = f"results/figures/ind_spec_const_{feature_to_highlight}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":
    # plot_alt_spec_features(save_fig=True)
    plot_ind_spec_constant(save_fig=True)
