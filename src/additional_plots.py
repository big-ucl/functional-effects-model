"""
KDE-overlay plots of functional (taste) values for grouped alt-specific
features (e.g. all "travel time" features, all "cost" features), compared
across datasets (SwissMetro, LPMC) and models (FIS-GBDT, FIS-DNN).

This builds on `additional_plots_for_lpmc_sm`: same model-loading and
functional-value computation/clipping logic, but instead of one
histogram-per-feature-per-model figure, it overlays KDEs of every feature
in a named group on a single axis, arranged in a 2x2 grid:

    rows    = dataset   (SwissMetro, LPMC)
    columns = model     (FIS-GBDT, FIS-DNN)

One such 2x2 figure is produced per feature group (e.g. one for
"travel_time", one for "cost").

Assumes functional_params=True and functional_intercept=True (i.e. the
FIS-GBDT / FIS-DNN variants), since that's what was requested. Only the
per-feature (non-intercept) functional values are needed for this plot.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler


from models_wrapper import RUMBoost, TasteNet
from constants import alt_spec_features, PATH_TO_DATA, PATH_TO_DATA_TRAIN
from utils import pkl_to_df
from rumboost.datasets import load_preprocess_LPMC


# These are assumed to already exist in your codebase (same as in the
# original `additional_plots_for_lpmc_sm`):
#   - pkl_to_df
#   - load_preprocess_LPMC
#   - lpmc_monotonic_constraints
#   - all_models, alt_spec_features, PATH_TO_DATA, PATH_TO_DATA_TRAIN
# from your_module import pkl_to_df, load_preprocess_LPMC, lpmc_monotonic_constraints


# ---------------------------------------------------------------------------
# 1. Fill these in with your actual column names / paths.
# ---------------------------------------------------------------------------

# Full, ordered list of alt-specific features used at training time, per
# dataset (same order as passed to `additional_plots_for_lpmc_sm` originally).
ALT_SPEC_FEATURES_PER_DATASET: Dict[str, List[str]] = {
    "SwissMetro": ["TRAIN_TT", "SM_TT", "CAR_TT", "TRAIN_CO", "SM_CO", "CAR_CO"],  # e.g. ["TRAIN_TT", "SM_TT", "CAR_TT", "TRAIN_CO", "SM_CO", "CAR_CO", "SM_HE", ...]
    "LPMC": ["dur_walking", "dur_cycling", "dur_pt_rail", "dur_driving", "cost_transit", "cost_driving_fuel"],        # e.g. ["dur_walking", "dur_cycling", "dur_pt_total", "dur_driving", ...]
}

# Data paths per dataset.
PATHS_PER_DATASET: Dict[str, Dict[str, str]] = {
    "SwissMetro": {"path_to_data": PATH_TO_DATA["SwissMetro"], "path_to_data_train": PATH_TO_DATA_TRAIN["SwissMetro"]},
    "LPMC": {"path_to_data": PATH_TO_DATA["LPMC"], "path_to_data_train": PATH_TO_DATA_TRAIN["LPMC"]},
}

# Which alt-specific features (by name, must match ALT_SPEC_FEATURES_PER_DATASET)
# belong to which group, per dataset. Add/rename groups as needed.
FEATURE_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "travel_time": {
        "SwissMetro": ["TRAIN_TT", "SM_TT", "CAR_TT"],  # e.g. ["TRAIN_TT", "SM_TT", "CAR_TT"]
        "LPMC": ["dur_walking", "dur_cycling", "dur_pt_rail", "dur_driving"],        # e.g. ["dur_walking", "dur_cycling", "dur_pt_total", "dur_driving"]
    },
    "cost": {
        "SwissMetro": ["TRAIN_CO", "SM_CO", "CAR_CO"],  # e.g. ["TRAIN_CO", "SM_CO", "CAR_CO"]
        "LPMC": ["cost_transit", "cost_driving_fuel"],        # e.g. ["cost_transit", "cost_driving_fuel"]
    },
}

MODEL_TITLES = {"RUMBoost": "FIS-GBDT", "TasteNet": "FIS-DNN"}
DATASET_ORDER = ["SwissMetro", "LPMC"]
MODEL_ORDER = ["RUMBoost", "TasteNet"]

FEATURE_COLORS = [
    "#004577", "#f54f1c", "#41def7", "#ff9500",
    "#7a2c8e", "#2ca02c", "#8c564b", "#e377c2",
]

TEX_FONTS = {
    "axes.labelsize": 7,
    "axes.linewidth": 0.5,
    "axes.labelpad": 1,
    "font.size": 7,
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


def _load_dataset(dataset: str, path_to_data: str, path_to_data_train: str, alt_spec_features: List[str]):
    """Reproduces the dataset-loading + socio-demo scaling branch from
    `additional_plots_for_lpmc_sm`."""
    if dataset == "SwissMetro":
        df = pkl_to_df(path_to_data)
        df_train = pkl_to_df(path_to_data_train)
        socio_demo_chars = [c for c in df.columns if c not in alt_spec_features and c not in ["CHOICE"]]
    elif dataset == "LPMC":
        df, _, _ = load_preprocess_LPMC(path_to_data)
        df_train = df
        socio_demo_chars = [c for c in df.columns if c not in alt_spec_features and c not in ["choice", "household_id"]]
    else:
        df = pd.read_csv(path_to_data)
        df_train = pd.read_csv(path_to_data_train)
        socio_demo_chars = [
            c for c in df.columns
            if c not in alt_spec_features and c not in ["mergeid", "hhid", "coupleid", "depression_scale"]
        ]

    scaler = MinMaxScaler()
    df_train[socio_demo_chars] = scaler.fit_transform(df_train[socio_demo_chars])
    df[socio_demo_chars] = scaler.transform(df[socio_demo_chars])
    return df, socio_demo_chars


def _clip_functional_values(y: np.ndarray, dataset: str, j: int) -> np.ndarray:
    """Reproduces the dataset-specific monotonic-constraint clipping from
    `additional_plots_for_lpmc_sm` (functional_params=True, functional_intercept=True,
    restricted to the feature part, i.e. j indexes into alt_spec_features directly)."""
    if dataset == "SwissMetro":
        if j != 6:
            y = np.minimum(y, 0)
    elif dataset == "LPMC":
        if j in lpmc_monotonic_constraints:
            y = np.minimum(y, 0)
    return y


def _compute_functional_values(
    dataset: str,
    alt_spec_features: List[str],
    all_models: Dict,
    path_to_data: str,
    path_to_data_train: str,
    needed_features: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads models for `dataset` and computes the (scaled, clipped) functional
    values for each feature in `needed_features`, for each model in `all_models`.

    Returns
    -------
    { feature_name: { model_name: np.ndarray of functional values } }
    """
    df, socio_demo_chars = _load_dataset(dataset, path_to_data, path_to_data_train, alt_spec_features)

    rumboost_predictor = None
    y_tastenet_full = None

    for model in all_models.keys():
        if model == "RUMBoost":
            model_path = f"results/{dataset}/{model}/model_fiTrue_fpTrue.json"
            rumboost = all_models[model]()
            rumboost.load_model(model_path)
            # last block of boosters corresponds to features + intercept classes;
            # we only need the feature part here, which comes first.
            num_classes = 1 if dataset == "easySHARE" else 3 if dataset == "SwissMetro" else 4
            num_plots = len(alt_spec_features) + num_classes
            rumboost_predictor = rumboost.model.boosters[-num_plots:]
        elif model == "TasteNet":
            model_path = f"results/{dataset}/{model}/model_fiTrue_fpTrue.pth"
            tastenet = all_models[model]()
            tastenet.load_model(path=model_path)
            tastenet_predictor = tastenet.model.params_module
            sdc_tensor = (
                torch.from_numpy(df[socio_demo_chars].values).to(torch.device("cuda")).to(torch.float32)
            )
            y_tastenet_full = tastenet_predictor(sdc_tensor).detach().cpu().numpy().squeeze()

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for feature in needed_features:
        j = alt_spec_features.index(feature)
        x_max = df[[feature]].max().values
        results[feature] = {}

        if "RUMBoost" in all_models:
            y_rumboost = rumboost_predictor[j].predict(df[socio_demo_chars]) / x_max
            y_rumboost = _clip_functional_values(y_rumboost, dataset, j)
            results[feature]["RUMBoost"] = y_rumboost

        if "TasteNet" in all_models:
            y_tastenet = y_tastenet_full[:, j] / x_max
            y_tastenet = _clip_functional_values(y_tastenet, dataset, j)
            results[feature]["TasteNet"] = y_tastenet

    return results


def plot_kde_overlay_by_group(
    feature_groups: Dict[str, Dict[str, List[str]]] = FEATURE_GROUPS,
    alt_spec_features_per_dataset: Dict[str, List[str]] = ALT_SPEC_FEATURES_PER_DATASET,
    paths_per_dataset: Dict[str, Dict[str, str]] = PATHS_PER_DATASET,
    all_models: Dict = None,
    save_fig: bool = False,
):
    """
    For each group in `feature_groups` (e.g. "travel_time", "cost"), produce a
    2x2 figure: rows = dataset (SwissMetro, LPMC), columns = model
    (FIS-GBDT, FIS-DNN). Each subplot overlays a KDE per feature in that
    group/dataset, all on the same x-axis (functional value scale).
    """
    sns.set_theme(font_scale=1, rc=TEX_FONTS)
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.family": "serif"})

    # Cache computed functional values per (dataset) so we don't reload
    # models/data once per group if a dataset's features appear in multiple groups.
    cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    def get_values(dataset, features):
        key = dataset
        if key not in cache:
            cache[key] = {}
        missing = [f for f in features if f not in cache[key]]
        if missing:
            computed = _compute_functional_values(
                dataset=dataset,
                alt_spec_features=alt_spec_features_per_dataset[dataset],
                all_models=all_models,
                path_to_data=paths_per_dataset[dataset]["path_to_data"],
                path_to_data_train=paths_per_dataset[dataset]["path_to_data_train"],
                needed_features=missing,
            )
            cache[key].update(computed)
        return {f: cache[key][f] for f in features}

    for group_name, per_dataset_features in feature_groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=300, sharex=False)

        for row, dataset in enumerate(DATASET_ORDER):
            features = per_dataset_features.get(dataset, [])
            if not features:
                for col in range(2):
                    axes[row, col].set_visible(False)
                continue

            values_by_feature = get_values(dataset, features)

            for col, model in enumerate(MODEL_ORDER):
                ax = axes[row, col]
                for i, feature in enumerate(features):
                    y = values_by_feature[feature].get(model)
                    if y is None:
                        continue
                    sns.kdeplot(
                        y,
                        ax=ax,
                        color=FEATURE_COLORS[i % len(FEATURE_COLORS)],
                        label=feature_names[feature],
                        fill=False,
                    )
                ax.set_title(f"{dataset} — {MODEL_TITLES[model]}")
                if col == 0:
                    ax.set_ylabel("Density")
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("Functional value")
                ax.legend(loc="best", frameon=True)

        fig.suptitle(f"Functional value distributions — {group_name.replace('_', ' ')}")
        fig.tight_layout()

        if save_fig:
            save_path = f"results/figures/kde_overlay_{group_name}_fiTrue_fpTrue.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Example call — fill in the placeholders above first.
    plot_kde_overlay_by_group(all_models=all_models, save_fig=True)