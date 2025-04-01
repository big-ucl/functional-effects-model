import pandas as pd
import torch

from utils import (
    generate_general_params,
    generate_rum_structure,
    generate_ordinal_spec,
    add_hyperparameters,
    build_lgb_dataset,
    split_dataset,
)
from rumboost.rumboost import rum_train
from constants import PATH_TO_DATA, alt_spec_features


def train(args):
    """
    Train the rumboost model.
    """

    # load the data
    data = pd.read_csv(PATH_TO_DATA)

    features = [
        col
        for col in data.columns
        if col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]
    target = "depression_scale"

    # split data
    X_train, y_train, X_test, y_test, folds = split_dataset(
        data,
        target,
        features,
        train_size=args.train_size,
        val_size=args.val_size,
        groups=data["hhid"],
        random_state=args.seed,
    )

    socio_demo_chars = [
        col
        for col in data.columns
        if col not in alt_spec_features
        and col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]

    # generate rum structure
    rum_structure = generate_rum_structure(alt_spec_features, socio_demo_chars)

    # generate ordinal spec
    ordinal_spec = generate_ordinal_spec(
        model_type=args.model_type, optim_interval=args.optim_interval
    )

    # generate general params
    general_params = generate_general_params(
        num_classes=13,
        num_iterations=args.num_iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
    )

    # add hyperparameters
    hyperparameters = {
        "num_leaves": args.num_leaves,
        "min_gain_to_split": args.min_gain_to_split,
        "min_sum_hessian_in_leaf": args.min_sum_hessian_in_leaf,
        "learning_rate": args.learning_rate,
        "max_bin": args.max_bin,
        "min_data_in_bin": args.min_data_in_bin,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
    }
    rum_structure[-1] = add_hyperparameters(rum_structure[-1], hyperparameters)

    model_spec = {
        "rum_structure": rum_structure,
        "general_params": general_params,
        "ordinal_logit": ordinal_spec,
    }

    # build lgb dataset
    lgb_train = build_lgb_dataset(X_train, y_train)
    lgb_test = build_lgb_dataset(X_test, y_test)

    # using gpu or not
    # if torch.cuda.is_available():
    #     torch_tensors = {"device": "cuda"}
    torch_tensors = None

    # train rumboost model
    model = rum_train(
        lgb_train,
        model_spec,
        valid_sets=[lgb_test],
        torch_tensors=torch_tensors,
    )
