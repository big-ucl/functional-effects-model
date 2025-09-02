import random
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Union

import lightgbm as lgb
import pandas as pd
import numpy as np

# type hints for dataset split function
TrainTestSplit = Union[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],  # train and test
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],  # train, val and test
]


def generate_rum_structure(
    alt_spec_features: Optional[Dict[str, List[str]]] = None,
    socio_demo_chars: Optional[List[str]] = None,
    functional_intercept: Optional[bool] = False,
    functional_params: Optional[bool] = False,
) -> List[Dict[str, Any]]:
    """
    Generate the rum structure for the given dataset. Note that this code is written for a single alternative (i.e. regression or ordinal regression problem).

    Parameters
    ----------
    alt_spec_features: Optional[Dict[str, List[str]]]
        The alternative-specific features to be used in the rum structure. The dictionary keys are the utility indices and the values are the features to be used in the rum structure.
    socio_demo_chars: Optional[List[str]]
        The socio-demographic characteristics to be used in the rum structure. They will represent the individual-specific constant learnt from the data.
    functional_intercept: Optional[bool]
        Whether to use the functional intercept or not. The default is False.
    functional_params: Optional[bool]
        Whether to use the functional parameters or not. The default is False.

    Returns
    -------
    rum_structure: List[Dict[Any]]
        The rum structure for the dataset.

    """

    # initialise rum_structure
    rum_structure = []

    # alternative-specific features, one per ensemble
    if not functional_params:
        for key, value in alt_spec_features.items():
            # monotone constraints for SwissMetro dataset
            if "TRAIN_TT" in value:
                monotone_constraints = [-1, -1, -1]
                md = 1
            elif "SM_TT" in value:
                monotone_constraints = [-1, -1, -1, 0]
                md = 1
            elif "CAR_TT" in value:
                monotone_constraints = [-1, -1]
                md = 1
            elif "dur_walking" in value:
                monotone_constraints = [-1, -1, 0, 0]
                md = 1
            elif "dur_cycling" in value:
                monotone_constraints = [-1, -1, 0, 0]
                md = 1
            elif "dur_pt_access" in value:
                monotone_constraints = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]
                md = 1
            elif "dur_driving" in value:
                monotone_constraints = [-1, -1, -1, -1, -1, 0, 0]
                md = 1
            elif value in [["f4"], ["f5"], ["f6"], ["f7"]]:
                monotone_constraints = [-1]
                md = -1
            else:
                monotone_constraints = [0] * len(value)
                md = 1
            interaction_constraints = [list(range(len(value)))]
            rum_structure_as = [
                {
                    "variables": value,
                    "utility": [key],
                    "boosting_params": {
                        "monotone_constraints_method": "advanced",
                        "max_depth": md,
                        "n_jobs": -1,
                        "learning_rate": 0.1,
                        "verbose": -1,
                        "monotone_constraints": monotone_constraints,
                        "interaction_constraints": interaction_constraints,
                    },
                    "shared": False,
                }
            ]
            # add the alternative-specific features to the rum_structure
            rum_structure.extend(rum_structure_as)
    else:
        # if functional parameters are used, add them to the rum_structure
        for key, value in alt_spec_features.items():
            if "TRAIN_TT" in value:
                monotone_constraints = [-1, -1, -1]
            elif "SM_TT" in value:
                monotone_constraints = [-1, -1, -1, 0]
            elif "CAR_TT" in value:
                monotone_constraints = [-1, -1]
            elif "dur_walking" in value:
                monotone_constraints = [-1, -1, 0, 0]
            elif "dur_cycling" in value:
                monotone_constraints = [-1, -1, 0, 0]
            elif "dur_pt_access" in value:
                monotone_constraints = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]
            elif "dur_driving" in value:
                monotone_constraints = [-1, -1, -1, -1, 0, 0, -1]
            else:
                monotone_constraints = [0] * len(value)
            rum_structure_params = [
                {
                    "variables": socio_demo_chars,
                    "utility": [key],
                    "boosting_params": {
                        "monotone_constraints_method": "advanced",
                        "n_jobs": -1,
                        "learning_rate": 0.1,
                        "monotone_constraints": [monotone_constraints[i]],
                        "verbose": -1,
                    },
                    "shared": False,
                    "endogenous_variable": f,
                }
                for i, f in enumerate(value)
            ]
            # add the functional parameters to the rum_structure
            rum_structure.extend(rum_structure_params)

    # socio-demographic characteristics, all in one ensemble
    if functional_intercept:
        for key, _ in alt_spec_features.items():
            rum_structure_sd = [
                {
                    "variables": socio_demo_chars,
                    "utility": [key],
                    "boosting_params": {
                        "monotone_constraints_method": "advanced",
                        "n_jobs": -1,
                        "learning_rate": 0.1,
                        "verbose": -1,
                    },
                    "shared": False,
                }
            ]

            # add the socio-demographic characteristics to the rum_structure
            rum_structure.extend(rum_structure_sd)

    return rum_structure


def add_hyperparameters(
    rum_struct: List[Dict[str, Any]],
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add hyperparameters to a specific dict of rum structure.

    Parameters
    ----------
    rum_struct: List[Dict[str, Any]]
        The rum structure to be modified.
    hyperparameters: Dict[str, Any]
        The hyperparameters to be added to the rum structure.

    Returns
    -------
    rum_structure: List[Dict[Any]]
        The modified rum structure with the hyperparameters added.

    """
    for struct in rum_struct:
        # add the hyperparameters to the rum structure
        struct["boosting_params"].update(hyperparameters)

    return rum_struct


def generate_general_params(num_classes: int, **kwargs) -> Dict[str, Any]:
    """ "
    Generate the general parameters for the rumboost model.

    Parameters
    ----------
    num_classes: int
        The number of classes in the dataset.
    kwargs: Dict[str, Any]
        The additional parameters to be added to the general parameters.
        These parameters will be used to update the general parameters.
        It has to be parameters that are accepted by rumboost.
        See the rumboost documentation for more details.

    Returns
    -------
    general_params: Dict[str, Any]
        The general parameters for the rumboost model.
    """
    # general parameters
    general_params = {
        "num_classes": num_classes,
        "max_booster_to_update": 1,
    }

    # update the general parameters with the kwargs
    general_params.update(kwargs)

    return general_params


def generate_ordinal_spec(
    model_type: Optional[str] = "proportional_odds", optim_interval: Optional[int] = 20
) -> Dict[str, Any]:
    """
    Generate the ordinal specification for the rumboost model.

    Parameters
    ----------
    model_type: str
        The type of the model. It can be either 'proportional_odds', 'coral' or 'corn'.
        The default is 'proportional_odds'.
    optim_interval: int
        The optimisation interval at which thresholds are updated with scipy. The default is 20.

    Returns
    -------
    ordinal_spec: Dict[str, Any]
        The ordinal specification for the rumboost model.
    """

    ordinal_spec = {
        "model": model_type,
        "optim_interval": optim_interval,
    }

    return ordinal_spec


def build_lgb_dataset(X: pd.DataFrame, y: pd.Series) -> lgb.Dataset:
    """
    Build the LightGBM dataset from the dataframe.

    Parameters
    ----------
    X: pd.DataFrame
        The dataframe to be used.
    y: pd.Series
        The target variable.

    Returns
    -------
    lgb_dataset: Any
        The LightGBM dataset.
    """

    # create the LightGBM dataset
    lgb_dataset = lgb.Dataset(X, label=y, free_raw_data=False)

    return lgb_dataset


def split_dataset(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    train_size: float = 0.8,
    val_size: Optional[float] = None,
    random_state: int = 42,
    groups: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> TrainTestSplit:
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to be used.
    target: str
        The target variable.
    features: List[str]
        The features to be used.
    train_size: float
        The size of the training set. The default is 0.7. This is the fraction of the total dataset.
    val_size: Optional[float]
        The size of the validation set. The default is 0.1. This is the fraction of the total dataset.
    random_state: int
        The random state to be used. The default is 42.
    groups: Optional[pd.Series]
        Whether to use stratified sampling or not. The default is None.
        If None, the data will be split randomly. If not None, the data will be split
        using the groups provided.
    save_path: Optional[str]
        The path to save the train and test sets. The default is None.

    Returns
    -------
    TrainTestSplit
        The train and test sets. If val_size is provided, the train set will be split into
        train and validation sets.
        If groups is provided, the train and test sets will be split using stratified sampling.
        If groups is not provided, the train and test sets will be split randomly.
    """
    assert (
        train_size + (val_size if val_size else 0) <= 0.95
    ), "The sum of train and val size must be less than 0.95."
    assert (
        train_size + (val_size if val_size else 0) >= 0.5
    ), "The sum of train and val size must be greater than 0.5"

    test_size = 1 - train_size - (val_size if val_size else 0)

    if groups is None:
        # if no groups are provided, use random sampling
        train_df = df.sample(frac=train_size, random_state=random_state).reset_index(
            drop=True
        )
        test_df = df.drop(train_df.index).reset_index(drop=True)

        if save_path:
            train_df.to_csv(save_path + "train.csv", index=False)
            test_df.to_csv(save_path + "test.csv", index=False)

        if val_size:
            val_size = val_size / (1 - test_size)
            train_df = train_df.sample(
                frac=val_size, random_state=random_state
            ).reset_index(drop=True)
            val_df = train_df.drop(train_df.index).reset_index(drop=True)
            if save_path:
                val_df.to_csv(save_path + "val.csv", index=False)

            return (
                train_df[features],
                train_df[target],
                val_df[features],
                val_df[target],
                test_df[features],
                test_df[target],
            )

        return train_df[features], train_df[target], test_df[features], test_df[target]

    # split the data into train and test sets
    folds = stratified_group_k_fold(
        df[features],
        df[target],
        groups,
        k=int(1 / test_size),
        seed=random_state,
    )
    for train_idx, test_idx in folds:
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        groups_train = groups.iloc[train_idx].reset_index(drop=True)
        break

    if save_path:
        train_df.to_csv(save_path + "train.csv", index=False)
        test_df.to_csv(save_path + "test.csv", index=False)

    if val_size:
        val_size = val_size / (1 - test_size)
        val_folds = stratified_group_k_fold(
            train_df[features],
            train_df[target],
            groups_train,
            k=int(1 / val_size),
            seed=random_state,
        )
        if save_path:
            pickle.dump(
                list(val_folds),
                open(
                    save_path + "folds.pickle",
                    "wb",
                ),
            )
        for train_idx, val_idx in val_folds:
            val_df = train_df.iloc[val_idx].reset_index(drop=True)
            train_df = train_df.iloc[train_idx].reset_index(drop=True)
            break

        return (
            train_df[features],
            train_df[target],
            val_df[features],
            val_df[target],
            test_df[features],
            test_df[target],
        )

    return train_df[features], train_df[target], test_df[features], test_df[target]


# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    """
    Stratified Group K-Fold cross-validator
    Provides train/test indices to split data in train/test sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into train/test set.
    k : int
        Number of folds. Must be at least 2.
    seed : int, optional
        Random seed for shuffling the data.

    Yields
    ------
    train : ndarray
        The training set indices for that split.
    test : ndarray
        The testing set indices for that split.
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def compute_metrics(
    preds: np.ndarray,
    binary_preds: np.ndarray,
    labels: np.ndarray,
    y_test: pd.Series,
) -> tuple[float, float, float]:
    """
    Compute the metrics for the model.

    Parameters
    ----------
    preds: np.ndarray
        The predictions of the model.
    binary_preds: np.ndarray
        The binary predictions of the model.
    labels: np.ndarray
        The labels of the model.
    y_test: pd.Series
        The test set.

    Returns
    -------
    mae_test: float
        The mean absolute error of the model.
    loss_test: float
        The loss of the model.
    emae_test: float
        The expected mean absolute error of the model.
    """
    mae_test = np.mean(np.abs(labels - y_test.values))

    safe_binary_preds = np.clip(binary_preds, 1e-15, 1 - 1e-7)

    ranks = np.arange(binary_preds.shape[1])
    levels = y_test.values[:, None] > ranks[None, :]
    loss_test = -np.mean(
        levels * np.log(safe_binary_preds)
        + (1 - levels) * np.log(1 - safe_binary_preds),
        axis=1,
    ).mean()

    all_labels = np.arange(preds.shape[1])
    distances = np.abs(all_labels[None, :] - y_test.values[:, None])
    emae_test = np.mean(preds * distances, axis=1).mean()

    return mae_test, loss_test, emae_test


def cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the cross entropy loss.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted labels.

    Returns
    -------
    loss: float
        The cross entropy loss.
    """
    indices = range(len(y_true))
    return -np.mean(np.log(y_pred[indices, y_true]))


def pkl_to_df(pkl_path: str) -> pd.DataFrame:
    """
    Convert a pickle file to a pandas dataframe.
    Parameters
    ----------
    pkl_path: str
        The path to the pickle file.
    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the data from the pickle file.
    """
    data = pd.read_pickle(pkl_path)
    data_df = pd.DataFrame(data["x"], columns=data["x_names"])
    data_df[data["z_names"]] = data["z"]
    data_df["CHOICE"] = data["y"] - 1
    data_df["CHOICE"] = data_df["CHOICE"].astype(int)
    data = data_df.copy()
    return data
