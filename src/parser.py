import argparse

def parse_cmdline_args(raw_args=None, parser=None):

    if parser is None:
        parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "--outpath",
        type=str,
        default="",
        help="Output path for the model",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="coral",
        help="Ordinal model type: proportional_odds or coral",
        choices=["proportional_odds", "coral"],
    )

    parser.add_argument(
        "--optim_interval",
        type=int,
        default=20,
        help="Optimisation interval for the ordinal model",
    )

    parser.add_argument(
        "--num_leaves",
        type=int,
        default=31,
        help="Number of leaves in the tree",
    )
    parser.add_argument(
        "--min_gain_to_split",
        type=float,
        default=0.0,
        help="Minimum gain to split the tree",
    )
    parser.add_argument(
        "--min_sum_hessian_in_leaf",
        type=float,
        default=0.001,
        help="Minimum sum hessian in leaf",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the model",
    )
    parser.add_argument(
        "--max_bin",
        type=int,
        default=255,
        help="Maximum number of bins for the model",
    )
    parser.add_argument(
        "--min_data_in_bin",
        type=int,
        default=3,
        help="Minimum number of data in bin",
    )
    parser.add_argument(
        "--min_data_in_leaf",
        type=int,
        default=20,
        help="Minimum number of data in leaf",
    )
    parser.add_argument(
        "--feature_fraction",
        type=float,
        default=1.,
        help="Feature fraction for the model",
    )
    parser.add_argument(
        "--bagging_fraction",
        type=float,
        default=1.,
        help="Bagging fraction for the model",
    )
    parser.add_argument(
        "--bagging_freq",
        type=int,
        default=0,
        help="Bagging frequency for the model",
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=0.0,
        help="L1 regularisation for the model",
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=0.0,
        help="L2 regularisation for the model",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3000,
        help="Number of iterations for the model",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=10,
        help="Early stopping rounds for the model",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of threads for the model",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity of the model",
    )
    parser.add_argument(
        "--verbose_interval",
        type=int,
        default=10,
        help="Verbosity interval of the model evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the model",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.64,
        help="Train size for the model",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.16,
        help="Validation size for the model",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="false",
        help="Save the model to disk",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for the model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs for the model",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=16,
        help="Number of layers for the model",
    )
    parser.add_argument(
        "--layer_sizes",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Layer sizes for the model",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for the model",
    )
    parser.add_argument(
        "--batch_norm",
        type=str,
        default="false",
        help="Batch normalisation for the model",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--act_func",
        type=str,
        default="relu",
        help="Activation function for the model",
        choices=["relu", "tanh", "sigmoid"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RUMBoost",
        required=True,
        help="Model to train",
        choices=["RUMBoost", "ResLogit", "TasteNet"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
        choices=["cuda", "cpu"],
    )


    parser.set_defaults(feature=True)
    args = parser.parse_args(raw_args)

    d = {'true': True,
         'false': False}

    args.save_model = d[args.save_model]  
    args.batch_norm = d[args.batch_norm]

    return args
