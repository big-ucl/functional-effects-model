import torch
import pandas as pd
import lightgbm as lgb
import numpy as np
import copy
from torch.utils.data import DataLoader

from utils import (
    generate_general_params,
    generate_rum_structure,
    generate_ordinal_spec,
    add_hyperparameters,
    build_lgb_dataset,
)
from rumboost.rumboost import rum_train
from rumboost.rumboost import RUMBoost as load_rumboost

from reslogit.models import OrdinalResLogit
from reslogit.data_utils import ResLogitDataset

from tastenet.models import TasteNet as TasteNetBuild
from tastenet.data_utils import TasteNetDataset


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
            num_classes=kwargs.get("num_classes", 13),
            num_iterations=kwargs.get("args").num_iterations,
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
        if kwargs.get("args").device == "cuda":
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

        self.best_iteration = self.model.best_iteration

        return self.model.best_score_train, self.model.best_score

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.

        Returns
        -------
        preds : np.array
            Predicted target variable, as probabilities.
        binary_preds : np.array
            The binary probabilities of the target being bigger than each level.
        label_pred : np.array
            Predicted target variable, as labels.
        """
        assert hasattr(
            self, "model"
        ), "Model not trained yet. Please train the model before predicting."
        # build lgb dataset
        lgb_test = lgb.Dataset(X_test, free_raw_data=False)
        preds = self.model.predict(lgb_test)
        if self.torch_tensors:
            preds = preds.cpu().numpy()
        binary_preds = -np.cumsum(preds, axis=1)[:, :-1] + 1
        label_preds = np.sum(binary_preds > 0.5, axis=1)
        return preds, binary_preds, label_preds

    def save_model(self, path: str):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        assert hasattr(
            self, "model"
        ), "Model not trained yet. Please train the model before saving."
        self.model.save_model(path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model = load_rumboost(model_file=path)


class ResLogit:
    """
    Wrapper class for Ordinal ResLogit model.
    """

    def __init__(self, **kwargs):
        self.alt_spec_features = kwargs.get("alt_spec_features")
        self.socio_demo_chars = kwargs.get("socio_demo_chars")

        self.model = OrdinalResLogit(
            kwargs.get("num_classes", 13),
            self.alt_spec_features,
            self.socio_demo_chars,
            kwargs.get("args").n_layers,
            kwargs.get("args").batch_size,
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
            patience=self.patience / 2,
            verbose=True,
        )

        self.device = torch.device(kwargs.get("args").device)
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
            self.alt_spec_features,
            self.socio_demo_chars,
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
                self.alt_spec_features,
                self.socio_demo_chars,
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
        self.model.train()

        best_loss = 1e10
        best_val_loss = 1e10
        patience_counter = 0

        for epoch in range(self.num_epochs):
            train_loss = 0

            for i, (x, y, z) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                z = z.to(self.device)
                classes = torch.arange(self.model.n_choices - 1).to(self.device)
                levels = (y[:, None] > classes[None, :]).float()

                self.optimiser.zero_grad()

                output = self.model(x, z)  # binary logits
                loss = self.criterion(output, levels)
                loss.backward()
                self.optimiser.step()

                train_loss += loss.item()
                if i % 50 == 0:
                    print(
                        f"--- Batch {i}/{len(self.train_dataloader)}, loss: {loss.item():.4f}"
                    )
            train_loss /= len(self.train_dataloader)

            if self.valid_dataloader is not None:
                val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for i, (x, y, z) in enumerate(self.valid_dataloader):
                        x = x.to(self.device)
                        y = y.to(self.device)
                        z = z.to(self.device)
                        classes = torch.arange(self.model.n_choices - 1).to(self.device)
                        levels = (y[:, None] > classes[None, :]).float()

                        output = self.model(x, z)  # binary logits
                        val_loss += self.criterion(output, levels).item()
                val_loss /= len(self.valid_dataloader)
                self.scheduler.step(val_loss)
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}: train loss = {train_loss:.4f}, val. loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss = train_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_iteration = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping")
                        break
        
        if hasattr(self, "best_model"):
            self.model = self.best_model

        return best_loss, best_val_loss

    def save_model(self, path: str):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : lgb.Dataset
            Test set.

        Returns
        -------
        preds : np.array
            Predicted target variable, as probabilities.
        binary_preds : np.array
            The binary probabilities of the target being bigger than each level.
        label_pred : np.array
            Predicted target variable, as labels.
        """
        self.model.eval()
        x = (
            torch.from_numpy(X_test.loc[:, self.alt_spec_features].values)
            .to(torch.float32)
            .to(self.device)
        )
        z = (
            torch.from_numpy(X_test.loc[:, self.socio_demo_chars].values)
            .to(torch.float32)
            .to(self.device)
        )
        logits = self.model(x, z)
        binary_preds = torch.sigmoid(logits)
        label_pred = torch.sum(binary_preds > 0.5, axis=1)
        preds = -torch.diff(
            binary_preds,
            dim=1,
            prepend=torch.ones(x.shape[0], device=self.device)[:, None],
            append=torch.zeros(x.shape[0], device=self.device)[:, None],
        )

        return preds.detach().cpu().numpy(), binary_preds.detach().cpu().numpy(), label_pred.detach().cpu().numpy()


class TasteNet:
    """
    Wrapper class for TasteNet model.
    """

    def __init__(self, **kwargs):

        self.alt_spec_features = kwargs.get("alt_spec_features")
        self.socio_demo_chars = kwargs.get("socio_demo_chars")
        self.num_classes = kwargs.get("num_classes", 13)

        self.model = TasteNetBuild(
            kwargs.get("args"),
            len(self.alt_spec_features),
            len(self.socio_demo_chars),
            self.num_classes,
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
            patience=self.patience / 2,
            verbose=True,
        )
        self.device = torch.device(kwargs.get("args").device)
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
        self.train_dataset = TasteNetDataset(
            X_train,
            y_train,
            self.alt_spec_features,
            self.socio_demo_chars,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if X_valid is not None and y_valid is not None:
            self.valid_dataset = TasteNetDataset(
                X_valid,
                y_valid,
                self.alt_spec_features,
                self.socio_demo_chars,
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
        self.model.train()

        best_loss = 1e10
        best_val_loss = 1e10
        patience_counter = 0

        for epoch in range(self.num_epochs):
            train_loss = 0

            for i, (x, y, z) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                z = z.to(self.device)
                classes = torch.arange(self.num_classes - 1).to(self.device)
                levels = (y[:, None] > classes[None, :]).float()

                self.optimiser.zero_grad()
                output = self.model(x, z)  # binary logits
                loss = self.criterion(output, levels)
                loss.backward()
                self.optimiser.step()

                train_loss += loss.item()
                if i % 50 == 0:
                    print(
                        f"--- Batch {i}/{len(self.train_dataloader)}, loss: {loss.item():.4f}"
                    )
            train_loss /= len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {train_loss}")

            if self.valid_dataloader is not None:
                val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for i, (x, y, z) in enumerate(self.valid_dataloader):
                        x = x.to(self.device)
                        y = y.to(self.device)
                        z = z.to(self.device)
                        classes = torch.arange(self.num_classes - 1).to(self.device)
                        levels = (y[:, None] > classes[None, :]).float()

                        output = self.model(x, z)  # binary logits
                        val_loss += self.criterion(output, levels).item()
                val_loss /= len(self.valid_dataloader)
                self.scheduler.step(val_loss)
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}: train loss = {train_loss:.4f}, val. loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss = train_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_iteration = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping")
                        break

        if hasattr(self, "best_model"):
            self.model = self.best_model

        return best_loss, best_val_loss

    def save_model(self, path: str):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.

        Returns
        -------
        preds : np.array
            Predicted target variable, as probabilities.
        binary_preds : np.array
            The binary probabilities of the target being bigger than each level.
        label_pred : np.array
            Predicted target variable, as labels.
        """
        self.model.eval()
        x = (
            torch.from_numpy(X_test.loc[:, self.alt_spec_features].values)
            .to(torch.float32)
            .to(self.device)
        )
        z = (
            torch.from_numpy(X_test.loc[:, self.socio_demo_chars].values)
            .to(torch.float32)
            .to(self.device)
        )
        logits = self.model(x, z)
        binary_preds = torch.sigmoid(logits)
        label_pred = torch.sum(binary_preds > 0.5, axis=1)
        preds = -torch.diff(
            binary_preds,
            dim=1,
            prepend=torch.ones(x.shape[0], device=self.device)[:, None],
            append=torch.zeros(x.shape[0], device=self.device)[:, None],
        )

        return preds.detach().cpu().numpy(), binary_preds.detach().cpu().numpy(), label_pred.detach().cpu().numpy()
