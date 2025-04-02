import torch
import pandas as pd
import lightgbm as lgb
import numpy as np
from torch.utils.data import DataLoader

from utils import (
    generate_general_params,
    generate_rum_structure,
    generate_ordinal_spec,
    add_hyperparameters,
    build_lgb_dataset,
)
from rumboost.rumboost import rum_train

from reslogit.models import OrdinalResLogit
from reslogit.data_utils import ResLogitDataset


class RUMBoost:
    """
    Wrapper class for RUMBoost model.
    """

    def __init__(self, **kwargs):
        # generate rum structure
        self.rum_structure = generate_rum_structure(
            kwargs.get("alt_spec_features"), kwargs.get("socio_demo_chars")
        )

        # generate ordinal spec
        ordinal_spec = generate_ordinal_spec(
            model_type=kwargs.get("args").model_type,
            optim_interval=kwargs.get("args").optim_interval,
        )

        # generate general params
        general_params = generate_general_params(
            num_classes=13,
            num_iterations=kwargs.get["args"].num_iterations,
            early_stopping_rounds=kwargs.get("args").early_stopping_rounds,
            verbose=kwargs.get("args").verbose,
            verbose_interval=kwargs.get("args").verbose_interval,
        )


        # add hyperparameters
        hyperparameters = {
            "num_leaves": kwargs.get("args").num_leaves,
            "min_gain_to_split": kwargs.get("args").min_gain_to_split,
            "min_sum_hessian_in_leaf": kwargs.get("args").min_sum_hessian_in_leaf,
            "learning_rate": kwargs.get("args").learning_rate,
            "max_bin": kwargs.get("args").max_bin,
            "min_data_in_bin": kwargs.get("args").min_data_in_bin,
            "min_data_in_leaf": kwargs.get("args").min_data_in_leaf,
            "feature_fraction": kwargs.get("args").feature_fraction,
            "bagging_fraction": kwargs.get("args").bagging_fraction,
            "bagging_freq": kwargs.get("args").bagging_freq,
            "lambda_l1": kwargs.get("args").lambda_l1,
            "lambda_l2": kwargs.get("args").lambda_l2,
        }
        self.rum_structure[-1] = add_hyperparameters(
            self.rum_structure[-1], hyperparameters
        )

        self.model_spec = {
            "rum_structure": self.rum_structure,
            "general_params": general_params,
            "ordinal_logit": ordinal_spec,
        }

        # using gpu or not
        if torch.cuda.is_available():
            self.torch_tensors = {"device": "cuda"}
        else:
            self.torch_tensors = None

    def build_dataloader(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """
        Builds and stores the LightGBM dataset. 
        There is no specific dataloader for RUMBoost.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target variable.
        X_valid : pd.DataFrame, optional
            Validation features. The default is None.
        y_valid : pd.Series, optional
            Validation target variable. The default is None.
        """
        self.lgb_train = build_lgb_dataset(
            X_train,
            y_train,
        )
        if X_valid is not None and y_valid is not None:
            self.lgb_valid = build_lgb_dataset(
                X_valid,
                y_valid,
            )

    def fit(self):
        """
        Fits the model to the training data.
        """
        # train rumboost model
        self.model = rum_train(
            self.lgb_train,
            self.model_spec,
            valid_sets=[self.lgb_valid] if hasattr(self, "lgb_valid") else None,
            torch_tensors=self.torch_tensors,
        )

        return self.model.best_score_train, self.model.best_score

    def predict(self, X_test: lgb.Dataset) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : lgb.Dataset
            Test set.

        Returns
        -------
        np.array
            Predicted target variable, as probabilities.
        """
        return self.model.predict(X_test)


class ResLogit:
    """
    Wrapper class for Ordinal ResLogit model.
    """

    def __init__(self, **kwargs):
        self.model = OrdinalResLogit(
            input=kwargs.get("input"),
            choice=kwargs.get("choice"),
            n_vars=kwargs.get("n_vars"),
            n_choices=kwargs.get("n_choices"),
            n_layers=kwargs.get("n_layers"),
        )

        self.batch_size = kwargs.get("args").batch_size
        self.num_epochs = kwargs.get("args").num_epochs
        self.patience = kwargs.get("args").patience

        self.optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=kwargs.get("args").learning_rate,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="min",
            factor=0.5,
            patience=self.patience,
            verbose=True,
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)


    def build_dataloader(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """
        Builds and stores the LightGBM dataset.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target variable.
        X_valid : pd.DataFrame, optional
            Validation features. The default is None.
        y_valid : pd.Series, optional
            Validation target variable. The default is None.
        """
        self.train_dataset = ResLogitDataset(
            X_train,
            y_train,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if X_valid is not None and y_valid is not None:
            self.valid_dataset = ResLogitDataset(
                X_valid,
                y_valid,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            self.valid_dataloader = None
            self.valid_dataset = None


    def fit(self):
        """
        Fits the model to the training data.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            best_loss = 1e10
            best_val_loss = 1e10

            for i, (x, y) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                classes = torch.arange(self.model.n_choices - 1).to(self.device)
                levels = y[:, None] > classes[None, :]

                self.optimiser.zero_grad()
                self.model.fit()
                output = self.model.output #binary logits
                loss = self.criterion(output, levels)
                loss.backward()
                self.optimiser.step()

                train_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Batch {i}, Loss: {loss.item()}")
            train_loss /= len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {train_loss}")

            if self.valid_dataloader is not None:
                val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for x, y in self.valid_dataloader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        classes = torch.arange(self.model.n_choices - 1).to(self.device)
                        levels = y[:, None] > classes[None, :]

                        output = self.model.output
                        val_loss += self.criterion(output, levels)
                val_loss /= len(self.valid_dataloader)
                self.scheduler.step(val_loss)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss.item()}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss = train_loss

        return best_loss, best_val_loss

    def predict(self, X_test: lgb.Dataset) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : lgb.Dataset
            Test set.

        Returns
        -------
        np.array
            Predicted target variable, as probabilities.
        """
        return self.model.predict(X_test)
