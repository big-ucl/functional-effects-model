import torch
import pickle
import pandas as pd
import lightgbm as lgb
import numpy as np
import copy
from torch.utils.data import DataLoader
from biogeme.calculator.single_formula import calculate_single_formula_from_expression
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.expressions import MonteCarlo
import biogeme.database as db
import biogeme.biogeme as bio

from utils import (
    generate_general_params,
    generate_rum_structure,
    generate_ordinal_spec,
    add_hyperparameters,
    build_lgb_dataset,
)
from rumboost.rumboost import rum_train
from rumboost.rumboost import RUMBoost as load_rumboost

from tastenet.models import TasteNet as TasteNetBuild
from tastenet.data_utils import TasteNetDataset

from machine_learning.dnn import DNN as DNNModel
from machine_learning.data_utils import DNNDataset

from statistical_models.mixed_effects import define_and_return_biogeme


class RUMBoost:
    """
    Wrapper class for RUMBoost model.
    """

    def __init__(self, **kwargs):
        # generate rum structure
        if (
            kwargs.get("alt_spec_features") is not None
            or kwargs.get("socio_demo_chars") is not None
        ):
            # if alt_spec_features or socio_demo_chars are not None, generate rum structure
            self.rum_structure = generate_rum_structure(
                kwargs.get("alt_spec_features"),
                kwargs.get("socio_demo_chars"),
                kwargs.get("args").functional_intercept,
                kwargs.get("args").functional_params,
            )

        if "args" in kwargs:
            # calculate number of boosters
            num_alt_spec_boosters = 0
            num_utility = 0
            min_utility = np.inf
            for key, value in kwargs.get("alt_spec_features").items():
                num_alt_spec_boosters += len(value)
                num_utility += 1
                if len(value) < min_utility:
                    min_utility = len(value)

            num_boosters = kwargs.get(
                "args"
            ).functional_intercept * num_utility + np.maximum(
                kwargs.get("args").functional_params * num_alt_spec_boosters,
                num_utility,
            )
            # generate ordinal spec
            if kwargs.get("args").dataset == "easySHARE":
                ordinal_spec = generate_ordinal_spec(
                    model_type=kwargs.get("args").model_type,
                    optim_interval=kwargs.get("args").optim_interval,
                )
            else:
                ordinal_spec = {}

            # generate general params
            general_params = generate_general_params(
                num_classes=kwargs.get("num_classes", 13),
                num_iterations=kwargs.get("args").num_iterations,
                early_stopping_rounds=kwargs.get("args").early_stopping_rounds,
                verbose=kwargs.get("args").verbose,
                verbose_interval=kwargs.get("args").verbose_interval,
                objective=(
                    "regression" if kwargs.get("num_classes") == 1 else "multiclass"
                ),
            )

            if kwargs.get("args").functional_params:
                boost_from_param_space = [True] * num_boosters
                if kwargs.get("args").functional_intercept:
                    boost_from_param_space[-num_utility:] = [False] * num_utility

                general_params["boost_from_parameter_space"] = boost_from_param_space

            max_boosters = (
                np.maximum(
                    num_utility * min_utility * kwargs.get("args").functional_params,
                    num_utility,
                )
                + num_utility * kwargs.get("args").functional_intercept
            )
            general_params["max_booster_to_update"] = max_boosters

            # add hyperparameters
            hyperparameters = {
                "num_leaves": kwargs.get("args").num_leaves,
                "min_gain_to_split": kwargs.get("args").min_gain_to_split,
                "min_sum_hessian_in_leaf": kwargs.get("args").min_sum_hessian_in_leaf,
                "learning_rate": np.minimum(
                    kwargs.get("args").learning_rate / max_boosters,
                    0.1,
                ),
                "max_bin": kwargs.get("args").max_bin,
                "min_data_in_bin": kwargs.get("args").min_data_in_bin,
                "min_data_in_leaf": kwargs.get("args").min_data_in_leaf,
                "feature_fraction": kwargs.get("args").feature_fraction,
                "bagging_fraction": kwargs.get("args").bagging_fraction,
                "bagging_freq": kwargs.get("args").bagging_freq,
                "lambda_l1": kwargs.get("args").lambda_l1,
                "lambda_l2": kwargs.get("args").lambda_l2,
                # "objective": "regression" if kwargs.get("num_classes") == 1 else "binary",
            }
            self.rum_structure[-num_boosters:] = add_hyperparameters(
                self.rum_structure[-num_boosters:], hyperparameters
            )

            if kwargs.get("args").dataset == "easySHARE":
                self.model_spec = {
                    "rum_structure": self.rum_structure,
                    "general_params": general_params,
                    "ordinal_logit": ordinal_spec,
                }
            else:
                self.model_spec = {
                    "rum_structure": self.rum_structure,
                    "general_params": general_params,
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

    def predict(self, X_test: pd.DataFrame, utilities: bool = False) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.
        utilities : bool
            Whether to predict utilities or probabilities.
            If True, returns raw utility values.

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
        preds = self.model.predict(lgb_test, utilities=utilities)
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


class TasteNet:
    """
    Wrapper class for TasteNet model.
    """

    def __init__(self, **kwargs):

        if "args" in kwargs:
            self.alt_spec_features = kwargs.get("alt_spec_features")
            all_alt_spec_features = []
            self.utility_structure = {}
            for i, (key, value) in enumerate(self.alt_spec_features.items()):
                self.utility_structure[key] = (
                    len(all_alt_spec_features),
                    len(all_alt_spec_features) + len(value),
                )
                all_alt_spec_features.extend(value)
            self.alt_spec_features = all_alt_spec_features

            self.socio_demo_chars = kwargs.get("socio_demo_chars")
            self.num_classes = kwargs.get("num_classes", 13)
            self.num_latent_vals = kwargs.get("num_latent_vals", 1)

            self.dataset = kwargs.get("args").dataset

            self.model = TasteNetBuild(
                kwargs.get("args"),
                len(self.alt_spec_features),
                len(self.socio_demo_chars),
                self.num_classes,
                self.num_latent_vals,
                self.utility_structure,
            )

            self.batch_size = kwargs.get("args").batch_size
            self.num_epochs = kwargs.get("args").num_epochs
            self.patience = kwargs.get("args").patience
            self.l1_norm = kwargs.get("args").lambda_l1
            self.l2_norm = kwargs.get("args").lambda_l2
            self.verbose = kwargs.get("args").verbose

            self.functional_params = kwargs.get("args").functional_params
            self.functional_intercept = kwargs.get("args").functional_intercept

            self.optimiser = torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get("args").learning_rate,
            )
            self.criterion = (
                torch.nn.BCEWithLogitsLoss()
                if kwargs.get("args").dataset == "easySHARE"
                else (
                    torch.nn.MSELoss()
                    if self.num_classes == 1
                    else torch.nn.CrossEntropyLoss()
                )
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimiser,
                mode="min",
                factor=0.5,
                patience=self.patience / 2,
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

                if self.dataset == "easySHARE":
                    classes = torch.arange(self.num_classes - 1).to(self.device)
                    levels = (y[:, None] > classes[None, :]).float()
                else:
                    levels = y

                self.optimiser.zero_grad()

                output = self.model(x, z)  # logits
                loss = self.criterion(output, levels)
                train_loss += loss.item()

                if self.l1_norm > 0 and (
                    self.functional_params or self.functional_intercept
                ):
                    loss += self.l1_norm * self.model.l1_norm().item() / x.shape[0]
                if self.l2_norm > 0 and (
                    self.functional_params or self.functional_intercept
                ):
                    loss += self.l2_norm * self.model.l2_norm().item() / x.shape[0]

                loss.backward()
                self.optimiser.step()

                if self.verbose > 0 and i % 50 == 0:
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
                        if self.dataset == "easySHARE":
                            classes = torch.arange(self.num_classes - 1).to(self.device)
                            levels = (y[:, None] > classes[None, :]).float()
                        else:
                            levels = y

                        output = self.model(x, z)  # binary logits
                        val_loss += self.criterion(output, levels).item()
                val_loss /= len(self.valid_dataloader)
                self.scheduler.step(val_loss)
                if self.verbose > 0:
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
            Path to save the model. Extension should be .pth.
        """
        torch.save(self.model, path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model = torch.load(path, weights_only=False)
        self.model.eval()

    def predict(self, X_test: pd.DataFrame, utilities: bool = False) -> np.array:
        """
        Predicts the target variable for the test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.
        utilities : bool
            Whether to predict utilities or not.

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
        if utilities:
            return logits.detach().cpu().numpy(), None, None
        if self.dataset == "easySHARE":
            binary_preds = torch.sigmoid(logits)
            label_pred = torch.sum(binary_preds > 0.5, axis=1)
            preds = -torch.diff(
                binary_preds,
                dim=1,
                prepend=torch.ones(x.shape[0], device=self.device)[:, None],
                append=torch.zeros(x.shape[0], device=self.device)[:, None],
            )
        else:
            preds = torch.softmax(logits, dim=1)
            label_pred = None
            binary_preds = None

        return (
            preds.detach().cpu().numpy(),
            binary_preds.detach().cpu().numpy() if binary_preds is not None else None,
            label_pred.detach().cpu().numpy() if label_pred is not None else None,
        )


class GBDT:
    """Wrapper class for GBDT model. Only implemented for classification and regression."""

    def __init__(self, **kwargs):
        if "args" in kwargs:
            if kwargs.get("args").early_stopping_rounds is not None:
                callback_es = lgb.early_stopping(
                    stopping_rounds=kwargs.get("args").early_stopping_rounds
                )
            else:
                callback_es = None
            self.model = lgb.LGBMClassifier(
                objective=(
                    "regression" if kwargs.get("num_classes", 13) == 1 else "multiclass"
                ),
                num_class=kwargs.get("num_classes", 13),
                num_leaves=kwargs.get("args").num_leaves,
                learning_rate=kwargs.get("args").learning_rate,
                n_estimators=kwargs.get("args").num_iterations,
                min_child_samples=kwargs.get("args").min_data_in_leaf,
                min_child_weight=kwargs.get("args").min_sum_hessian_in_leaf,
                min_split_gain=kwargs.get("args").min_gain_to_split,
                subsample=kwargs.get("args").bagging_fraction,
                subsample_freq=kwargs.get("args").bagging_freq,
                colsample_bytree=kwargs.get("args").feature_fraction,
                reg_alpha=kwargs.get("args").lambda_l1,
                reg_lambda=kwargs.get("args").lambda_l2,
                n_jobs=-1,
                verbose=kwargs.get("args").verbose,
                callbacks=[callback_es],
                importance_type="gain",
            )
            self.features = kwargs.get("alt_spec_features")
            self.num_classes = kwargs.get("num_classes", 13)

    def build_dataloader(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """
        Builds and stores the LightGBM dataset.
        There is no specific dataloader for GBDT.

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
        self.X_train = X_train[self.features]
        self.y_train = y_train
        if X_valid is not None and y_valid is not None:
            self.X_valid = X_valid[self.features]
            self.y_valid = y_valid

    def fit(self):
        """
        Fits the model to the training data.
        """
        if hasattr(self, "X_valid") and hasattr(self, "y_valid"):
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)],
            )
            best_score_train = self.model.best_score_["training"]["multi_logloss"]
            best_score_valid = self.model.best_score_["valid_1"]["multi_logloss"]
            self.best_iteration = self.model.best_iteration_
        else:
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_train, self.y_train)],
            )
            best_score_train = self.model.best_score_["training"]["multi_logloss"]
            best_score_valid = None
            self.best_iteration = self.model.n_estimators_

        return best_score_train, best_score_valid

    def predict(self, X_test: pd.DataFrame, utilities: bool = False) -> np.array:
        """ "
        Predicts the target variable for the test set."

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.
        utilities : bool
            Whether to predict utilities or probabilities.
            If True, returns raw utility values.

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

        preds = self.model.predict_proba(X_test[self.features], raw_score=utilities)

        # ordinal not needed for GBDT
        if self.num_classes == 2:
            preds = np.hstack((1 - preds.reshape(-1, 1), preds.reshape(-1, 1)))
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
        self.model.booster_.save_model(path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model.booster_ = lgb.Booster(model_file=path)


class DNN:
    """Wrapper class for DNN model. Only implemented for classification and regression."""

    def __init__(self, **kwargs):
        self.model = DNNModel(
            layers=kwargs.get("args").layer_sizes,
            activation=kwargs.get("args").act_func,
            dropout=kwargs.get("args").dropout,
            batch_norm=kwargs.get("args").batch_norm,
            input_dim=len(kwargs.get("alt_spec_features", [])),
            output_dim=kwargs.get("num_classes", 13),
            device=kwargs.get("args").device,
        )
        self.features = kwargs.get("alt_spec_features", [])
        self.num_classes = kwargs.get("num_classes", 13)
        self.num_epochs = kwargs.get("args").num_epochs
        self.patience = kwargs.get("args").patience
        self.device = torch.device(kwargs.get("args").device)
        self.verbose = kwargs.get("args").verbose
        self.learning_rate = kwargs.get("args").learning_rate
        self.l1_norm = kwargs.get("args").lambda_l1
        self.l2_norm = kwargs.get("args").lambda_l2
        self.batch_size = kwargs.get("args").batch_size
        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if kwargs.get("args").dataset == "easySHARE"
            else (
                torch.nn.MSELoss()
                if self.num_classes == 1
                else torch.nn.CrossEntropyLoss()
            )
        )

        self.model.to(self.device)
        self.optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="min",
            patience=self.patience / 2,
            factor=0.5,
        )

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
        self.train_dataset = DNNDataset(
            X_train,
            y_train,
            self.features,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if X_valid is not None and y_valid is not None:
            self.valid_dataset = DNNDataset(
                X_valid,
                y_valid,
                self.features,
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

            for i, (x, y) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                if (
                    self.num_classes == 2
                    and self.criterion.__class__ == torch.nn.BCEWithLogitsLoss
                ):
                    classes = torch.arange(self.num_classes - 1).to(self.device)
                    levels = (y[:, None] > classes[None, :]).float()
                else:
                    levels = y

                self.optimiser.zero_grad()

                output = self.model(x)  # logits
                loss = self.criterion(output, levels)
                train_loss += loss.item()

                if self.l1_norm > 0:
                    loss += self.l1_norm * self.model.l1_norm().item() / x.shape[0]
                if self.l2_norm > 0:
                    loss += self.l2_norm * self.model.l2_norm().item() / x.shape[0]

                loss.backward()
                self.optimiser.step()

                if self.verbose > 0 and i % 50 == 0:
                    print(
                        f"--- Batch {i}/{len(self.train_dataloader)}, loss: {loss.item():.4f}"
                    )
            train_loss /= len(self.train_dataloader)

            if self.valid_dataloader is not None:
                val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for i, (x, y) in enumerate(self.valid_dataloader):
                        x = x.to(self.device)
                        y = y.to(self.device)
                        if (
                            self.num_classes == 2
                            and self.criterion.__class__ == torch.nn.BCEWithLogitsLoss
                        ):
                            classes = torch.arange(self.num_classes - 1).to(self.device)
                            levels = (y[:, None] > classes[None, :]).float()
                        else:
                            levels = y

                        output = self.model(x)  # binary logits
                        val_loss += self.criterion(output, levels).item()
                val_loss /= len(self.valid_dataloader)
                self.scheduler.step(val_loss)
                if self.verbose > 0:
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
            Path to save the model. Extension should be .pth.
        """
        torch.save(self.model, path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model = torch.load(path, weights_only=False)
        self.model.eval()

    def predict(self, X_test: pd.DataFrame, utilities: bool = False) -> np.array:
        """
        Predicts the target variable for the test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.
        utilities : bool
            Whether to predict utilities or not.

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
            torch.from_numpy(X_test.loc[:, self.features].values)
            .to(torch.float32)
            .to(self.device)
        )
        logits = self.model(x)
        if utilities:
            return logits.detach().cpu().numpy(), None, None
        if (
            self.num_classes == 2
            and self.criterion.__class__ == torch.nn.BCEWithLogitsLoss
        ):
            binary_preds = torch.sigmoid(logits)
            label_pred = torch.sum(binary_preds > 0.5, axis=1)
            preds = -torch.diff(
                binary_preds,
                dim=1,
                prepend=torch.ones(x.shape[0], device=self.device)[:, None],
                append=torch.zeros(x.shape[0], device=self.device)[:, None],
            )
        else:
            preds = torch.softmax(logits, dim=1)
            label_pred = None
            binary_preds = None

        return (
            preds.detach().cpu().numpy(),
            binary_preds.detach().cpu().numpy() if binary_preds is not None else None,
            label_pred.detach().cpu().numpy() if label_pred is not None else None,
        )


class MixedEffect:
    """Wrapper class for Mixed Effect model."""

    def __init__(self, alt_spec_features: dict, socio_demo_chars: list, num_classes: int = 13, **kwargs):

        self.alt_spec_features = alt_spec_features
        self.socio_demo_chars = socio_demo_chars
        self.num_classes = num_classes

    def build_dataloader(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """
        Builds and stores the Biogeme database.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target variable.
        X_val : pd.DataFrame, optional
            Validation features. The default is None.
        y_val : pd.Series, optional
            Validation target variable. The default is None.
        """

        df = X_train.copy()
        df["CHOICE"] = y_train.values

        (
            self.model,
            self.log_probability,
            self.conditional_trajectory_probability,
            self.ascs,
            self.database,
        ) = define_and_return_biogeme(df, self.alt_spec_features, self.num_classes)

    def fit(self) -> tuple[float, None]:
        """
        Fits the model to the training data.
        """
        assert hasattr(
            self, "model"
        ), "Dataloader not built yet. Please build the dataloader before fitting the model."

        results = self.model.estimate()

        self.results = results
        self.params = results.get_estimated_parameters()
        self.betas = results.get_beta_values()
        self.loglike = results.final_log_likelihood

        return self.loglike, None

    def predict(self, X_test: pd.DataFrame, utilities: bool = False) -> np.array:
        """
        Predicts the target variable for the test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.
        utilities : bool
            Whether to predict utilities or not.

        Returns
        -------
        cel : np.array
            Cross-entropy loss on the test set.
        binary_preds : np.array
            The binary probabilities of the target being bigger than each level.
        label_pred : np.array
            Predicted target variable, as labels.
        """
        assert hasattr(
            self, "results"
        ), "Model not trained yet. Please train the model before predicting."

        df = X_test.copy()
        df["CHOICE"] = 1  # placeholder

        database = db.Database("test", df)
        database.panel("ID")

        simulated_loglike = calculate_single_formula_from_expression(
            expression=self.log_probability,
            database=database,
            number_of_draws=500,
            the_betas=self.results.get_beta_values(),
            numerically_safe=False,
            second_derivatives_mode=SecondDerivativesMode.NEVER,
            use_jit=True,
        )

        cel = simulated_loglike / df.shape[0]

        return cel, None, None

    def get_individual_parameters(self, on_train_set: bool) -> pd.DataFrame:
        """
        Returns the individual-specific parameters.

        Parameters
        ----------
        on_train_set : bool
            Whether to get the parameters for the training set or test set.

        Returns
        -------
        individual_params : pd.DataFrame
            DataFrame containing individual-specific parameters.
        """
        assert hasattr(
            self, "results"
        ), "Model not trained yet. Please train the model before getting individual parameters."

        if not on_train_set:
            ascs = self.params[self.params["Name"].str.contains("asc_")]["Value"]
            ascs = np.append(ascs, 0.0)  # last class asc is 0
            print(ascs)
            return ascs

        simulate = {}
        for i, asc in enumerate(self.ascs):
            if i != self.num_classes - 1:
                simulate["Numerator_" + str(i)] = MonteCarlo(
                    asc * self.conditional_trajectory_probability
                )
                simulate["Denominator_" + str(i)] = MonteCarlo(
                    self.conditional_trajectory_probability
                )

        biosim = bio.BIOGEME(self.database, simulate, number_of_draws=500, seed=1)

        sim = biosim.simulate(self.results.get_beta_values())

        individual_params = pd.DataFrame()
        for i, _ in enumerate(self.ascs):
            if i != self.num_classes - 1:
                individual_params["asc_" + str(i)] = sim["Numerator_" + str(i)] / sim["Denominator_" + str(i)]
            else:
                individual_params["asc_" + str(i)] = 0.0

        return individual_params.values
    
    def save_model(self, path: str) -> None:
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        assert hasattr(
            self, "results"
        ), "Model not trained yet. Please train the model before saving."
        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    def load_model(self, path: str) -> None:
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        with open(path, "rb") as f:
            self.results = pickle.load(f)
        self.params = self.results.get_estimated_parameters()
        self.betas = self.results.get_beta_values()
        self.loglike = self.results.data.logLike
