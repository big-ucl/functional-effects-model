import random
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Union, Generator

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
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        Generator[List[int], List[int]],
    ],  # train, test and folds
]


def generate_rum_structure(
    alt_spec_features: Optional[List[str]] = None,
    socio_demo_chars: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate the rum structure for the given dataset. Note that this code is written for a single alternative (i.e. regression or ordinal regression problem).

    Parameters
    ----------
    alt_spec_features: Optional[List[str]]
        The alternative-specific features to be used in the rum structure.
    socio_demo_chars: Optional[List[str]]
        The socio-demographic characteristics to be used in the rum structure. They will represent the individual-specific constant learnt from the data.

    Returns
    -------
    rum_structure: List[Dict[Any]]
        The rum structure for the dataset.

    """
    assert (
        alt_spec_features or socio_demo_chars
    ), "At least one alternative-specific feature or socio-demographic characteristics must be provided."

    # initialise rum_structure
    rum_structure = []

    # alternative-specific features, one per ensemble
    if alt_spec_features:
        rum_structure_as = [
            {
                "variables": [f],
                "utility": [0],
                "boosting_params": {
                    "monotone_constraints_method": "advanced",
                    "max_depth": 1,
                    "n_jobs": -1,
                    "learning_rate": 0.1,
                    "monotone_constraints": [
                        0,
                    ],
                    "interaction_constraints": [
                        [0],
                    ],
                },
                "shared": False
            }
            for f in alt_spec_features
        ]

        # add the alternative-specific features to the rum_structure
        rum_structure.extend(rum_structure_as)

    # socio-demographic characteristics, all in one ensemble
    if socio_demo_chars:
        rum_structure_sd = [
            {
                "variables": socio_demo_chars,
                "utility": [0],
                "boosting_params": {
                    "monotone_constraints_method": "advanced",
                    "n_jobs": -1,
                    "learning_rate": 0.1,
                    "monotone_constraints": [
                        0,
                    ]
                    * len(socio_demo_chars),
                    "interaction_constraints": [
                        list(range(len(socio_demo_chars))),
                    ],
                },
                "shared": False,
            }
            
        ]

        # add the socio-demographic characteristics to the rum_structure
        rum_structure.extend(rum_structure_sd)

    return rum_structure


def add_hyperparameters(
    struct: Dict[str, Any],
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add hyperparameters to a specific dict of rum structure.

    Parameters
    ----------
    struct: Dict[str, Any]
        The rum structure to be modified.
    hyperparameters: Dict[str, Any]
        The hyperparameters to be added to the rum structure.

    Returns
    -------
    rum_structure: List[Dict[Any]]
        The modified rum structure with the hyperparameters added.

    """
    # add the hyperparameters to the rum structure
    struct["boosting_params"].update(hyperparameters)

    return struct


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

        if val_size:
            val_size = val_size / (1 - test_size)
            train_df = train_df.sample(
                frac=val_size, random_state=random_state
            ).reset_index(drop=True)
            val_df = train_df.drop(train_df.index).reset_index(drop=True)
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
        break

    if val_size:
        val_size = val_size / (1 - test_size)
        folds = stratified_group_k_fold(
            train_df[features],
            train_df[target],
            groups,
            k=int(1 / val_size),
            seed=random_state,
        )

        return (
            train_df[features],
            train_df[target],
            test_df[features],
            test_df[target],
            folds,
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
