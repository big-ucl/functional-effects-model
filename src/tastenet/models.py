import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(nl_func):
    if nl_func == "tanh":
        return nn.Tanh()
    elif nl_func == "relu":
        return nn.ReLU()
    elif nl_func == "sigmoid":
        return nn.Sigmoid()
    else:
        return None


class TasteNet(nn.Module):
    """TasteNet-MNL model for Swissmetro"""

    def __init__(
        self, args, num_alt_features, num_sd_chars, num_classes, num_latent_vals=None
    ):
        """
        Initialize the TasteNet class.

        Args:
        args (argparse.Namespace): command line arguments.
        num_alt_features (int): number of alternative features.
        num_sd_chars (int): number of socio-demographic characteristics.
        num_classes (int): number of classes.
        num_latent_vals (int, optional): number of latent values. Defaults to None.
            Useful only for ordinal regression problems, since the number of latent vars is 1.
        """
        super(TasteNet, self).__init__()

        self.func_intercept = args.functional_intercept
        self.func_params = args.functional_params

        if not num_latent_vals:
            num_latent_vals = num_classes

        if self.func_intercept or self.func_params:
            self.params_module = TasteParams(
                args.layer_sizes,
                args,
                num_alt_features,
                num_latent_vals,
                num_sd_chars,
                self.func_intercept,
                self.func_params,
            )
        self.util_module = Utility(
            args,
            num_alt_features,
            num_latent_vals,
            self.func_intercept,
            self.func_params,
        )
        self.ordinal_module = Coral_layer(num_classes)
        self.args = args

    def forward(self, x, z=None):
        if self.func_intercept or self.func_params:
            b = self.params_module(z)  # taste parameters, (N,1)
        else:
            b = None
        v = self.util_module(x, b)  # no softmax here

        logits = self.ordinal_module(v)  # (N, J-1)

        return logits

    def l2_norm(self):
        """
        L2 norm, not including bias
        """
        norm = torch.zeros(1).to(device=torch.device(self.args.device))
        for i, params in enumerate(self.params_module.parameters()):
            if i % 2 == 1:
                # skip bias
                continue
            norm += (params**2).sum()

        return norm

    def l1_norm(self):
        """
        L1 norm, not including bias
        """
        norm = torch.zeros(1).to(device=torch.device(self.args.device))
        for i, params in enumerate(self.params_module.parameters()):
            if i % 2 == 1:
                # skip bias
                continue
            norm += torch.abs(params).sum()
            
        return norm


class Utility(nn.Module):
    def __init__(
        self, args, num_alt_features, num_classes, func_intercept=True, func_params=True
    ):
        super(Utility, self).__init__()
        self.args = args

        self.func_intercept = func_intercept
        self.func_params = func_params

        self.num_classes = num_classes
        self.num_alt_features = num_alt_features

        if not self.func_params:
            self.mnl = nn.Linear(num_alt_features, num_classes)

        if not self.func_intercept:
            self.intercept = nn.Parameter(torch.zeros(num_classes))  # (1, J)

    def forward(self, x, b=None):
        """
        x: attributes of each alternative,
           including the intercept (N,K)  J alternatives, each have K attributes.
        b: taste parameters (N, 1): Individual taste parameters.
        """
        if not self.func_params and not self.func_intercept:
            v = self.mnl(x) + self.intercept.view(
                1, self.num_classes
            )  # (N, J) + (1, J) = (N, J)
        elif self.func_params and not self.func_intercept:
            v = (
                x[:, :, None] * b.view(-1, self.num_alt_features, self.num_classes)
            ).sum(dim=1) + self.intercept.view(
                1, self.num_classes
            )  # (N, K, :) * (N, K, J).sum(dim=1) + (1, J)  = (N, J)
        elif not self.func_params and self.func_intercept:
            v = self.mnl(x) + b.view(-1, self.num_classes)  # (N, J) + (N, J) = (N, J)
        else:
            v = (
                x[:, :, None]
                * b.view(-1, self.num_alt_features + 1, self.num_classes)[:, :-1, :]
            ).sum(dim=1) + b.view(-1, self.num_alt_features + 1, self.num_classes)[
                :, -1, :
            ]  # (N, K+1, :) * (N, K+1, J) . sum(dim=1) = (N, J)

        return v


class TasteParams(nn.Module):
    """
    Network for tastes
    """

    def __init__(
        self,
        layer_sizes,
        args,
        num_alt_features,
        num_classes,
        num_sd_chars,
        func_intercept=True,
        func_params=True,
    ):
        """Initialize the TasteParams class.
        Args:
        layer_sizes (list[tuple]): list of layer sizes in a tuple.
        args (argparse.Namespace): command line arguments.
        func_intercept (bool): whether to include functional intercepts.
        func_params (bool): whether to include functional taste parameters.
        """
        if not func_intercept and not func_params:
            raise ValueError(
                "At least one of func_intercept or func_params must be True."
            )

        all_layers = [l for l in layer_sizes]
        all_layers.insert(0, num_sd_chars)
        all_layers.append(
            num_alt_features * num_classes * func_params + num_classes * func_intercept
        )

        super(TasteParams, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(all_layers[:-1], all_layers[1:])):
            self.seq.add_module(
                name=f"L{i+1}", module=nn.Linear(in_size, out_size, bias=True)
            )
            if i < len(layer_sizes) - 2:
                self.seq.add_module(name=f"A{i+1}", module=get_act(args.act_func))
                if args.dropout > 0:
                    self.seq.add_module(name=f"D{i+1}", module=nn.Dropout(args.dropout))
                if args.batch_norm:
                    self.seq.add_module(
                        name=f"BN{i+1}", module=nn.BatchNorm1d(out_size)
                    )
        self.args = args

    def forward(self, z):
        """
        Parameters:
            z: (N,D) # batch size, input dimension
        Returns:
            V: (N,1) # taste parameters
        """
        return self.seq(z)  # (N,K)


class Coral_layer(nn.Module):
    def __init__(self, n_choices):
        """Initialize the Ordinal_layer class (Coral layer).

        Args:
        n_choices (int): number of choice alternatives.
        """
        super(Coral_layer, self).__init__()
        self.coral_bias = nn.Parameter(torch.ones((n_choices - 1,)))

    def forward(self, x):
        """return the output of Coral layer.

        Args:
            input (TensorVariable):  output of last residual layer.
        """
        return x + self.coral_bias  # (N, J-1)
