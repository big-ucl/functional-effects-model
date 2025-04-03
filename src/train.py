import pandas as pd
import os
import time

from helper import set_all_seeds
from utils import split_dataset, compute_metrics
from constants import PATH_TO_DATA, alt_spec_features
from models_wrapper import RUMBoost, ResLogit, TasteNet


def train(args):
    """
    Train the specified model.
    """

    if not args.outpath:
        args.outpath = f"results/{args.model}/"

    # create the output directory if it does not exist
    os.makedirs(args.outpath, exist_ok=True)

    # set the random seed for reproducibility
    set_all_seeds(args.seed)

    # load the data
    data = pd.read_csv(PATH_TO_DATA)

    features = [
        col
        for col in data.columns
        if col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]
    target = "depression_scale"

    # split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
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

    if args.model == "RUMBoost":
        model = RUMBoost(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=13,
            args=args,
        )
        save_path = args.outpath + "model.json"
    elif args.model == "ResLogit":
        model = ResLogit(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=13,
            args=args,
        )
        save_path = args.outpath + "model.pkl"
    elif args.model == "TasteNet":
        model = TasteNet(
            alt_spec_features=alt_spec_features,
            socio_demo_chars=socio_demo_chars,
            num_classes=13,
            args=args,
        )
        save_path = args.outpath + "model.pkl"

    model.build_dataloader(X_train, y_train, X_val, y_val)

    # fit the model
    start_time = time.time()
    best_train_loss, best_val_loss = model.fit()
    end_time = time.time()

    # test the model
    preds, binary_preds, labels = model.predict(X_test)

    mae_test, loss_test, emae_test = compute_metrics(
        preds, binary_preds, labels, y_test
    )

    print(f"Best Train Loss: {best_train_loss}, Best Val Loss: {best_val_loss}")
    print(f"Test MAE: {mae_test}, Test Loss: {loss_test}, Test EMAE: {emae_test}")

    results_dict = {
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "train_time": end_time - start_time,
        "mae_test": mae_test,
        "loss_test": loss_test,
        "emae_test": emae_test,
    }

    # save the results
    pd.DataFrame(results_dict, index=[0]).to_csv(args.outpath + "results_dict.csv")

    if args.save_model:
        model.save_model(save_path)
