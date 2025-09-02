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
        self,
        args,
        num_alt_features,
        num_sd_chars,
        num_classes,
        num_latent_vals=None,
        utility_structure=None,
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
        utility_structure (str, optional): structure of the utility function. Defaults to None.
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
            utility_structure=utility_structure,
        )
        self.ordinal_module = Coral_layer(num_classes) if num_latent_vals == 1 else None
        self.args = args
        self.num_classes = num_classes

    def forward(self, x, z=None):
        if self.func_intercept or self.func_params:
            b = self.params_module(z)  # taste parameters, (N,1)
            if self.num_classes == 3 and self.func_params:
                b = self.monotonic_constraints(b)
            elif self.num_classes == 4 and self.func_params:
                b = self.lpmc_monotonic_constraints(b)
        else:
            b = None
        v = self.util_module(x, b)  # no softmax here

        if self.ordinal_module is None:
            logits = v
        else:
            logits = self.ordinal_module(v)  # (N, J-1)

        return logits

    def monotonic_constraints(self, b):
        """
        Put transformation for the sake of constraints on the value of times
        This is only for the SwissMetro dataset and needs to be adapted for other datasets.
        b: taste parameters (N, 1): Individual taste parameters.

        """
        if self.func_intercept:
            return torch.cat(
                [
                    -F.relu(-b[:, :6]),
                    b[:, 6].view(-1, 1),
                    -F.relu(-b[:, 7:9]),
                    b[:, -self.num_classes :].view(-1, self.num_classes),
                ],
                dim=1,
            )
        else:
            return torch.cat(
                [-F.relu(-b[:, :6]), b[:, 6].view(-1, 1), -F.relu(-b[:, 7:9])], dim=1
            )

    def lpmc_monotonic_constraints(self, b):
        """
        Put transformation for the sake of constraints on the value of times
        This is only for the LPMC dataset and needs to be adapted for other datasets.
        b: taste parameters (N, 1): Individual taste parameters.

        """
        if self.func_intercept:
            return torch.cat(
                [
                    -F.relu(-b[:, :2]),
                    b[:, 2:4].view(-1, 2),
                    -F.relu(-b[:, 4:6]),
                    b[:, 6:8].view(-1, 2),
                    -F.relu(-b[:, 8:16]),
                    b[:, 16:18].view(-1, 2),
                    -F.relu(-b[:, 18:23]),
                    b[:, 23:25].view(-1, 2),
                    b[:, -self.num_classes :].view(-1, self.num_classes),
                ],
                dim=1,
            )
        else:
            return torch.cat(
                [
                    -F.relu(-b[:, :2]),
                    b[:, 2:4].view(-1, 2),
                    -F.relu(-b[:, 4:6]),
                    b[:, 6:8].view(-1, 2),
                    -F.relu(-b[:, 8:16]),
                    b[:, 16:18].view(-1, 2),
                    -F.relu(-b[:, 18:23]),
                    b[:, 23:25].view(-1, 2),
                ],
                dim=1,
            )

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
        self,
        args,
        num_alt_features,
        num_classes,
        func_intercept=True,
        func_params=True,
        utility_structure=None,
    ):
        super(Utility, self).__init__()
        self.args = args

        self.func_intercept = func_intercept
        self.func_params = func_params

        self.num_classes = num_classes
        self.num_alt_features = num_alt_features
        self.utility_structure = utility_structure

        self.mnl = MNL_layer(utility_structure, args)

        if not self.func_intercept:
            self.intercept = nn.Parameter(torch.zeros(num_classes))  # (1, J)

    def forward(self, x, b=None):
        """
        x: attributes of each alternative,
           including the intercept (N,K)  J alternatives, each have K attributes.
        b: taste parameters (N, 1): Individual taste parameters.
        """
        if not self.func_params and not self.func_intercept:
            v = self.mnl(x) + self.intercept.view(1, self.num_classes)
        elif self.func_params and not self.func_intercept:
            v = self.mnl(x, b) + self.intercept.view(1, self.num_classes)
        elif not self.func_params and self.func_intercept:
            v = self.mnl(x) + b.view(-1, self.num_classes)
        else:
            v = self.mnl(x, b[:, : -self.num_classes]) + b[:, -self.num_classes :]

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
        num_alt_features (int): number of alternative features.
        num_classes (int): number of classes.
        num_sd_chars (int): number of socio-demographic characteristics.
        """
        if not func_intercept and not func_params:
            raise ValueError(
                "At least one of func_intercept or func_params must be True."
            )

        all_layers = [l for l in layer_sizes]
        all_layers.insert(0, num_sd_chars)
        all_layers.append(num_alt_features * func_params + num_classes * func_intercept)

        super(TasteParams, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(all_layers[:-1], all_layers[1:])):
            self.seq.add_module(
                name=f"L{i+1}", module=nn.Linear(in_size, out_size, bias=True)
            )
            if i < len(all_layers) - 2:
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


class MNL_layer(nn.Module):
    def __init__(self, utility_structure, args):
        """Initialize the MNL_complex_layer class.

        Args:
        n_choices (int): number of choice alternatives.
        args (argparse.Namespace): command line arguments.
        """
        super(MNL_layer, self).__init__()
        self.args = args
        self.mnl = nn.ModuleList()
        self.utility_structure = utility_structure
        for _, v in utility_structure.items():
            self.mnl.append(nn.Linear(v[1] - v[0], 1, bias=False))

    def forward(self, x, b=None):
        """return the output of MNL complex layer.

        Args:
            x (TensorVariable):  output of last residual layer.
            b (TensorVariable):  taste parameters.
        """
        logits = torch.zeros(x.shape[0], len(self.utility_structure)).to(
            device=torch.device(self.args.device)
        )
        for k, v in self.utility_structure.items():
            if b is not None:
                logits[:, k] = (x[:, v[0] : v[1]] * b[:, v[0] : v[1]]).sum(dim=1)
            else:
                logits[:, k] = self.mnl[k](x[:, v[0] : v[1]]).view(-1)
        return logits
