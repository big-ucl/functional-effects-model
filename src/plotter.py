import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os

from sklearn.preprocessing import MinMaxScaler

from models_wrapper import RUMBoost, TasteNet
from constants import alt_spec_features, PATH_TO_DATA, PATH_TO_DATA_TRAIN
from typing import List, Dict


all_models = {
    "RUMBoost": RUMBoost,
    "TasteNet": TasteNet,
}

feature_names = {
    "bmi": "BMI",
    "chronic_mod": "Number of chronic conditions",
    "daily_activities_index": "Daily activities index",
    "finemotor": "Fine motor skills",
    "grossmotor": "Gross motor skills",
    "hospitalised_last_year_yes": "Hospitalised last year",
    "lgmuscle": "Large muscle skills",
    "maxgrip": "Max grip strength",
    "mobilityind": "Mobility index",
    "nursing_home_last_year_yes_permanently": "Nursing home last year (permanently)",
    "nursing_home_last_year_yes_temporarily": "Nursing home last year (temporarily)",
    "recall_1": "Recall 1",
    "recall_2": "Recall 2",
    "sphus_excellent": "Self-perceived health - excellent",
    "sphus_fair": "Self-perceived health - fair",
    "sphus_good": "Self-perceived health - good",
    "sphus_poor": "Self-perceived health - poor",
    "sphus_very_good": "Self-perceived health - very good",
    "sphus_very_poor": "Self-perceived health - very poor",
    "instrumental_activities_index": "Instrumental activities index",
    "nb_doctor_visits": "Number of doctor visits",
}


def plot_alt_spec_features(
    alt_spec_features: List = alt_spec_features,
    all_models: Dict = all_models,
    path_to_data: str = PATH_TO_DATA,
    save_fig: bool = False,
):
    """
    Plot the alternative-specific features for the models, if trained without functional parameters.

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
            model_path_fi = f"results/{model}/model_fiTrue_fpFalse.json"
            rumboost_fi = all_models[model]()
            rumboost_fi.load_model(model_path_fi)
            rumboost_params_fi = rumboost_fi.model.boosters[:-1]

            model_path = f"results/{model}/model_fiFalse_fpFalse.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            rumboost_params = rumboost.model.boosters
        elif model == "TasteNet":
            model_path = f"results/{model}/model_fiFalse_fpFalse.pth"
            tastenet = all_models[model]()
            tastenet.load_model(path=model_path)
            tastenet_params = [
                params.detach().cpu().numpy()
                for params in tastenet.model.util_module.mnl.parameters()
            ][:-1][0]

            model_path_fi = f"results/{model}/model_fiTrue_fpFalse.pth"
            tastenet_fi = all_models[model]()
            tastenet_fi.load_model(path=model_path_fi)
            tastenet_params_fi = [
                params.detach().cpu().numpy()
                for params in tastenet_fi.model.util_module.mnl.parameters()
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

    colors = ["#264653", "#2a9d8f","#0073a1", "#7cd2bf"]

    for i, as_feat in enumerate(alt_spec_features):

        x = np.linspace(0, df[as_feat].max(), 10000)
        x_scaled = np.linspace(0, 1, 10000)
        dummy_array = np.zeros((10000, len(alt_spec_features)))
        dummy_array[:, i] = x_scaled
        y_rumboost = rumboost_params[0].predict(dummy_array)
        y_tastenet = tastenet_params[:, i] * x / x.max()

        y_rumboost_fi = rumboost_params_fi[0].predict(dummy_array)
        y_tastenet_fi = tastenet_params_fi[:, i] * x / x.max()

        y_rumboost = [y - y_rumboost[0] for y in y_rumboost]
        y_rumboost_fi = [y - y_rumboost_fi[0] for y in y_rumboost_fi]

        # Plot the features
        plt.figure(figsize=(2.62, 1.97), dpi=300)

        plt.plot(x, y_rumboost, label="RUMBoost", color=colors[0], linewidth=0.8)
        plt.plot(
            x, y_rumboost_fi, label="RUMBoost - FI", color=colors[2], linewidth=0.8
        )

        plt.plot(x, y_tastenet, label="TasteNet", color=colors[1], linewidth=0.8)
        plt.plot(
            x, y_tastenet_fi, label="TasteNet - FI", color=colors[3], linewidth=0.8
        )
        # plt.xlabel(feature_names[as_feat])
        plt.ylabel("Utility")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            save_path = f"results/figures/{feature_names[as_feat]}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # plt.show()


def plot_ind_spec_constant(
    alt_spec_features: List = alt_spec_features,
    all_models: Dict = all_models,
    path_to_data: str = PATH_TO_DATA,
    path_to_data_train: str = PATH_TO_DATA_TRAIN,
    save_fig: bool = False,
    feature_to_highlight: str = None,
    functional_params: bool = True,
    functional_intercept: bool = True,
):
    """
    Plot the individual-specific constant for the models.
    The model needs to be trained with functional parameters or functional intercept.

    Parameters
    ----------
    alt_spec_features : List
        List of alternative-specific features. They must be in the same order as for the training.
    all_models : Dict
        Dictionary of all models.
    path_to_data : str
        Path to the data folder.
    path_to_data_train : str
        Path to the training data folder.
    save_fig : bool
        Whether to save the figure or not.
    feature_to_highlight : str
        Feature to highlight in the plot. If None, no feature is highlighted.
    functional_params : bool
        If the model is trained with functional parameters.
    functional_intercept : bool
        If the model is trained with functional intercept.
    """

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
    df_train = pd.read_csv(path_to_data_train)

    socio_demo_chars = [
        col
        for col in df.columns
        if col not in alt_spec_features
        and col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]

    scaler = MinMaxScaler()
    df_train[socio_demo_chars] = scaler.fit_transform(df_train[socio_demo_chars])
    df[socio_demo_chars] = scaler.transform(df[socio_demo_chars])

    num_plots = functional_params * len(alt_spec_features) + functional_intercept
    if num_plots == 0:
        raise ValueError(
            "The model needs to be trained with functional parameters or functional intercept."
        )
    print(num_plots)
    # Load the models
    for model in all_models.keys():
        if model == "RUMBoost":
            model_path = f"results/{model}/model_fi{functional_intercept}_fp{functional_params}.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            rumboost_predictor = rumboost.model.boosters[-num_plots:]
        elif model == "TasteNet":
            model_path = f"results/{model}/model_fi{functional_intercept}_fp{functional_params}.pth"
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
                y_tastenet = y_tastenet.reshape(-1, 1)

    colors = ["#264653", "#2a9d8f","#0073a1", "#7cd2bf"]

    for j in range(num_plots):
        fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=300)
        if "RUMBoost" in all_models:
            y_rumboost = rumboost_predictor[j].predict(df[socio_demo_chars])
            min_val = y_rumboost.min()
            max_val = y_rumboost.max()
        if "TasteNet" in all_models:
            min_val = min(min_val, y_tastenet[:, j].min())
            max_val = max(max_val, y_tastenet[:, j].max())

        bin_edges = np.linspace(min_val, max_val, 50) 

        max_count = 0
        for model in all_models.keys():
            if model == "RUMBoost":
                counts, _ = np.histogram(y_rumboost, bins=bin_edges)
            elif model == "TasteNet":
                counts, _ = np.histogram(y_tastenet[:, j], bins=bin_edges)
            max_count = max(max_count, counts.max())

        for i, (model, ax) in enumerate(zip(all_models.keys(), axes.flatten())):
            if feature_to_highlight:
                df[feature_to_highlight] = df[feature_to_highlight].astype("category")
                df[feature_to_highlight] = df[feature_to_highlight].cat.codes
                hue = df[feature_to_highlight]
            else:
                hue = None
            if model == "RUMBoost":
                sns.histplot(
                    y_rumboost,
                    ax=ax,
                    bins=bin_edges,
                    color=colors[i],
                    hue=hue,
                    # multiple="stack",
                )
            elif model == "TasteNet":
                sns.histplot(
                    y_tastenet[:, j],
                    ax=ax,
                    bins=bin_edges,
                    color=colors[i],
                    hue=hue,
                    # multiple="stack",
                )

            title = model

            if functional_params and j < num_plots - 1:
                fig_title = f"{alt_spec_features[j]}"
            else:
                fig_title = "Intercept"
            ax.set_title(title)
            # ax.set_xlabel("Functional values")
            if i%2 == 0:
                ax.set_ylabel("Count")
                plt.title(fig_title, fontsize=8)
            else:
                ax.set_ylabel("")

            xlim = (min_val - (max_val - min_val)*0.01, max_val + (max_val - min_val)*0.01)
            ylim = (0, max_count * 1.1)

            plt.setp(axes, xlim=xlim, ylim=ylim)

        if save_fig:
            if not feature_to_highlight:
                feature_to_highlight = ""
            if functional_params and j < num_plots - 1:
                save_path = f"results/figures/ind_spec_const_{alt_spec_features[j]}_fi{functional_intercept}_fp{functional_params}_{feature_to_highlight}.png"
            else:
                save_path = f"results/figures/ind_spec_const_intercept_fiTrue_fp{functional_params}_{feature_to_highlight}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # plt.show()


if __name__ == "__main__":
    # plot_alt_spec_features(save_fig=True)
    plot_ind_spec_constant(save_fig=True)
    plot_ind_spec_constant(save_fig=True, functional_params=False, functional_intercept=True)
    plot_ind_spec_constant(save_fig=True, functional_params=True, functional_intercept=False)
    # plot_ind_spec_constant(functional_params=False, feature_to_highlight="female")

