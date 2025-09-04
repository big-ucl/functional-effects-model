import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os

from sklearn.preprocessing import MinMaxScaler

from models_wrapper import RUMBoost, TasteNet
from constants import alt_spec_features, PATH_TO_DATA, PATH_TO_DATA_TRAIN
from utils import pkl_to_df
from typing import List, Dict
from rumboost.datasets import load_preprocess_LPMC


all_models = {
    "RUMBoost": RUMBoost,
    "TasteNet": TasteNet,
}

feature_duplicated = ["distance", "day_of_week", "start_time_linear"]

lpmc_monotonic_constraints = [0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22]

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
    "SM_TT": "Swissmetro travel time",
    "SM_HE": "Swissmetro headway",
    "SM_CO": "Swissmetro cost",
    "SM_SEATS": "Swissmetro seats style",
    "TRAIN_TT": "Train travel time",
    "TRAIN_HE": "Train headway",
    "TRAIN_CO": "Train cost",
    "CAR_TT": "Car travel time",
    "CAR_CO": "Car cost",
    "dur_walking": "Walking travel time",
    "distance": "Distance",
    "day_of_week": "Day of the week",
    "start_time_linear": "Trip start time",
    "dur_cycling": "Cycling travel time",
    "dur_pt_access": "Public transport access time",
    "dur_pt_rail": "Public transport rail time",
    "dur_pt_bus": "Public transport bus time",
    "dur_pt_int_waiting": "Public transport interchanging waiting time",
    "dur_pt_int_walking": "Public transport interchanging walking time",
    "pt_n_interchanges": "Number of public transport interchanges",
    "cost_transit": "Public transport cost",
    "dur_driving": "Driving travel time",
    "cost_driving_fuel": "Driving fuel cost",
    "congestion_charge": "Congestion charge",
    "driving_traffic_percent": "Road congestion percentage",
}


def plot_alt_spec_features(
    alt_spec_features: List = alt_spec_features,
    all_models: Dict = all_models,
    path_to_data: str = PATH_TO_DATA,
    save_fig: bool = False,
    dataset: str = "SwissMetro",
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
    dataset : str
        Dataset to use. Default is "SwissMetro".
    """
    num_classes = 1 if dataset == "easySHARE" else 3 if dataset == "SwissMetro" else 4
    # Load the models
    for model in all_models.keys():
        if model == "RUMBoost":
            model_path_fi = f"results/{dataset}/{model}/model_fiTrue_fpFalse.json"
            rumboost_fi = all_models[model]()
            rumboost_fi.load_model(model_path_fi)
            rumboost_params_fi = rumboost_fi.model.boosters[:-num_classes]

            model_path = f"results/{dataset}/{model}/model_fiFalse_fpFalse.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            rumboost_params = rumboost.model.boosters
        elif model == "TasteNet":
            model_path = f"results/{dataset}/{model}/model_fiFalse_fpFalse.pth"
            tastenet = all_models[model]()
            tastenet.load_model(path=model_path)
            tastenet_params = np.array([])
            for param in tastenet.model.util_module.mnl.mnl.parameters():
                tastenet_params = np.concatenate(
                    [tastenet_params, param.detach().cpu().numpy()[0]]
                )

            model_path_fi = f"results/{dataset}/{model}/model_fiTrue_fpFalse.pth"
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
    if dataset == "SwissMetro":
        df = pkl_to_df(path_to_data)
    elif dataset == "LPMC":
        df, _, _ = load_preprocess_LPMC(path_to_data)
    else:
        df = pd.read_csv(path_to_data)

    colors = ["#264653", "#2a9d8f", "#0073a1", "#7cd2bf"]

    for i, as_feat in enumerate(alt_spec_features):

        x = np.linspace(0, df[as_feat].max(), 10000)
        x_scaled = np.linspace(0, 1, 10000)
        dummy_array = np.zeros((10000, len(alt_spec_features)))
        dummy_array[:, i] = x_scaled
        if len(alt_spec_features) == 9:
            k = 0 if i < 3 else 1 if i < 7 else 2
            l = 3 if i < 3 else 4 if i < 7 else 2
            m = 0 if i < 3 else 3 if i < 7 else 7
            y_rumboost = rumboost_params[k].predict(dummy_array[:, m : m + l])
        elif len(alt_spec_features) == 25:
            k = 0 if i < 4 else 1 if i < 8 else 2 if i < 18 else 3
            l = 4 if i < 4 else 4 if i < 8 else 10 if i < 18 else 7
            m = 0 if i < 4 else 4 if i < 8 else 8 if i < 18 else 18
            y_rumboost = rumboost_params[k].predict(dummy_array[:, m : m + l])
        else:
            y_rumboost = rumboost_params[0].predict(dummy_array)
        y_tastenet = tastenet_params[i] * x / x.max()

        if len(alt_spec_features) == 9:
            y_rumboost_fi = rumboost_params_fi[k].predict(dummy_array[:, m : m + l])
        elif len(alt_spec_features) == 25:
            y_rumboost_fi = rumboost_params_fi[k].predict(dummy_array[:, m : m + l])
        else:
            y_rumboost_fi = rumboost_params_fi[0].predict(dummy_array)
        y_tastenet_fi = tastenet_params_fi[i] * x / x.max()

        y_rumboost = [y - y_rumboost[0] for y in y_rumboost]
        y_rumboost_fi = [y - y_rumboost_fi[0] for y in y_rumboost_fi]

        # Plot the features
        plt.figure(figsize=(2.62, 1.97), dpi=300)

        plt.plot(x, y_rumboost, label="GBDT", color=colors[0], linewidth=0.8)
        plt.plot(
            x, y_rumboost_fi, label="GBDT - FI", color=colors[2], linewidth=0.8
        )

        plt.plot(x, y_tastenet, label="DNN", color=colors[1], linewidth=0.8)
        plt.plot(
            x, y_tastenet_fi, label="DNN - FI", color=colors[3], linewidth=0.8
        )
        # plt.xlabel(feature_names[as_feat])
        plt.ylabel("Utility")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            if dataset == "LPMC" and as_feat in feature_duplicated:
                class_number = 0 if i < 4 else 1 if i < 8 else 2 if i < 18 else 3
            else:
                class_number = ""
            save_path = f"results/{dataset}/figures/{feature_names[as_feat]}{class_number}.png"
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
    dataset: str = "SwissMetro",
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
    dataset : str
        Dataset to use. Default is "SwissMetro".
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

    if dataset == "SwissMetro":
        df = pkl_to_df(path_to_data)
        df_train = pkl_to_df(path_to_data_train)

        socio_demo_chars = [
            col
            for col in df.columns
            if col not in alt_spec_features and col not in ["CHOICE"]
        ]
    elif dataset == "LPMC":
        df, _, _ = load_preprocess_LPMC(path_to_data)
        df_train = df
        socio_demo_chars = [
            col
            for col in df.columns
            if col not in alt_spec_features and col not in ["choice"]
        ]
    else:
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

    num_classes = 1 if dataset == "easySHARE" else 3 if dataset == "SwissMetro" else 4

    num_plots = (
        functional_params * len(alt_spec_features) + functional_intercept * num_classes
    )
    if num_plots == 0:
        raise ValueError(
            "The model needs to be trained with functional parameters or functional intercept."
        )
    # Load the models
    for model in all_models.keys():
        if model == "RUMBoost":
            model_path = f"results/{dataset}/{model}/model_fi{functional_intercept}_fp{functional_params}.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            rumboost_predictor = rumboost.model.boosters[-num_plots:]

        elif model == "TasteNet":
            model_path = f"results/{dataset}/{model}/model_fi{functional_intercept}_fp{functional_params}.pth"
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
                y_tastenet = y_tastenet.reshape(-1, num_classes)
            if dataset == "easySHARE" and functional_intercept:
                first_threshold = (
                    tastenet.model.ordinal_module.coral_bias[0].detach().cpu().numpy()
                )
                y_tastenet[:, -1] = y_tastenet[:, -1] + first_threshold

    colors = ["#264653", "#2a9d8f", "#0073a1", "#7cd2bf"]

    for j in range(num_plots):
        fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=300)
        if functional_intercept:
            if j < num_plots - num_classes:
                x_max = df[[alt_spec_features[j]]].max().values
            else:
                x_max = 1
        else:
            x_max = df[[alt_spec_features[j]]].max().values
        if "RUMBoost" in all_models:
            y_rumboost = rumboost_predictor[j].predict(df[socio_demo_chars]) / x_max
            if dataset == "SwissMetro" and functional_intercept:
                if j != 6 and j < num_plots - num_classes:
                    y_rumboost = np.minimum(y_rumboost, 0)
            elif dataset == "SwissMetro" and j != 6:
                y_rumboost = np.minimum(y_rumboost, 0)

            if dataset == "LPMC" and functional_params and j in lpmc_monotonic_constraints:
                y_rumboost = np.minimum(y_rumboost, 0)

            if (
                dataset == "easySHARE"
                and functional_intercept
                and j >= num_plots - num_classes
            ):
                first_threshold = rumboost.model.thresholds[0]
                y_rumboost = y_rumboost - first_threshold
            min_val = y_rumboost.min()
            max_val = y_rumboost.max()
        if "TasteNet" in all_models:
            y_tastenet[:, j] = y_tastenet[:, j] / x_max
            if dataset == "SwissMetro" and functional_intercept: 
                if j != 6 and j < num_plots - num_classes:
                    y_tastenet[:, j] = np.minimum(y_tastenet[:, j], 0)
            elif dataset == "SwissMetro" and j != 6:
                y_tastenet[:, j] = np.minimum(y_tastenet[:, j], 0)

            if dataset == "LPMC" and functional_params and j in lpmc_monotonic_constraints:
                y_tastenet[:, j] = np.minimum(y_tastenet[:, j], 0)

            min_val = min(min_val, y_tastenet[:, j].min())
            max_val = max(max_val, y_tastenet[:, j].max())

        if min_val == max_val:
            min_val = min_val - 0.05
            max_val = max_val + 0.05

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

            title = "GBDT" if model == "RUMBoost" else "DNN"

            if functional_params and j < num_plots - num_classes:
                fig_title = f"{alt_spec_features[j]}"
            else:
                fig_title = "Intercept"
            ax.set_title(title)
            # ax.set_xlabel("Functional values")
            if i % 2 == 0:
                ax.set_ylabel("Count")
                plt.title(fig_title, fontsize=8)
            else:
                ax.set_ylabel("")

            xlim = (
                min_val - (max_val - min_val) * 0.01,
                max_val + (max_val - min_val) * 0.01,
            )
            ylim = (0, max_count * 1.1)

            plt.setp(axes, xlim=xlim, ylim=ylim)

        if save_fig:
            if not feature_to_highlight:
                feature_to_highlight = ""
            if dataset == "LPMC" and functional_params and j < len(alt_spec_features) and alt_spec_features[j] in feature_duplicated:
                class_number = 0 if j < 4 else 1 if j < 8 else 2 if j < 18 else 3
            else:
                class_number = ""
            if (
                functional_params
                and functional_intercept
                and j < num_plots - num_classes
            ):
                save_path = f"results/{dataset}/figures/ind_spec_const_{alt_spec_features[j]}{class_number}_fi{functional_intercept}_fp{functional_params}_{feature_to_highlight}.png"
            elif (
                functional_params
                and functional_intercept
                and j >= num_plots - num_classes
            ):
                save_path = f"results/{dataset}/figures/ind_spec_const_intercept_{j - len(alt_spec_features)}_fiTrue_fp{functional_params}_{feature_to_highlight}.png"
            elif functional_params and not functional_intercept:
                save_path = f"results/{dataset}/figures/ind_spec_const_{alt_spec_features[j]}{class_number}_fi{functional_intercept}_fp{functional_params}_{feature_to_highlight}.png"
            else:
                save_path = f"results/{dataset}/figures/ind_spec_const_intercept_{j}_fiTrue_fp{functional_params}_{feature_to_highlight}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # plt.show()


if __name__ == "__main__":

    for dataset in ["easySHARE", "LPMC", "SwissMetro"]:
        all_alt_spec_features = []
        for k, v in alt_spec_features[dataset].items():
            all_alt_spec_features.extend(v)
        path_to_data = PATH_TO_DATA[dataset]
        path_to_data_train = PATH_TO_DATA_TRAIN[dataset]
        if dataset in ["SwissMetro", "LPMC"]:
            path_to_data_train = path_to_data

        plot_alt_spec_features(
            all_alt_spec_features,
            path_to_data=path_to_data,
            save_fig=True,
            dataset=dataset,
        )
        plot_ind_spec_constant(
            all_alt_spec_features,
            path_to_data=path_to_data,
            path_to_data_train=path_to_data_train,
            save_fig=True,
            dataset=dataset,
        )
        plot_ind_spec_constant(
            all_alt_spec_features,
            path_to_data=path_to_data,
            path_to_data_train=path_to_data_train,
            save_fig=True,
            functional_params=False,
            functional_intercept=True,
            dataset=dataset,
        )
        plot_ind_spec_constant(
            all_alt_spec_features,
            path_to_data=path_to_data,
            path_to_data_train=path_to_data_train,
            save_fig=True,
            functional_params=True,
            functional_intercept=False,
            dataset=dataset,
        )
