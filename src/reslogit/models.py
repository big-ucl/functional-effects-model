import torch
import torch.nn as nn


class MNL(nn.Module):
    def __init__(self, n_vars, n_choices, bias=True):
        """Initialize the Logit class.

        Args:
            n_vars (int): number of input variables.
            n_choices (int): number of choice alternatives.
            bias (bool): whether to include bias in the model.
        """
        super(MNL, self).__init__()
        # define initial value for asc parameter and parameters associated to explanatory variables
        self.fc = nn.Linear(n_vars, n_choices, bias=bias)

    def forward(self, x):
        """return the output of Logit architecture.

        Args:
            x (TensorVariable):  input of Logit architecture.
        """
        self.output = self.fc(x)
        return self.output

class ResNetLayer(nn.Module):
    def __init__(self, n_in, n_out):
        """Initialize the ResNetLayer class.

        Args:
            input (TensorVariable)
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
        """
        super(ResNetLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        # define Initial value of residual layer weights
        W_init = torch.eye(self.n_out)

        # learnable parameters of a model
        self.W = nn.Parameter(W_init)
        self.params = [self.W]

    def forward(self, x):
        """return the output of each residual layer.

        Args:
            x (TensorVariable):  input of each residual layer.
        """
        self.lin_output = torch.matmul(x, self.W)

        output = x - nn.functional.softplus(self.lin_output)
        return output


class ResNet(nn.Module):
    def __init__(self, n_in, n_out, n_layers=16):
        """Initialize the ResNet architecture.

        Args:
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
            n_layers (int): number of residual layers.
        """
        super(ResNet, self).__init__()

        self.n_layers = n_layers

        # define n_layers residual layer
        self.layers = nn.Sequential(*[ResNetLayer(n_in, n_out) for _ in range(n_layers)])
        self.final_layer = nn.Linear(n_out, 1, bias=False)
        self.layers.add_module("final_layer", self.final_layer)

    def forward(self, x):
        """return the final output of ResNet architecture.

        Args:
            x (TensorVariable):  input of first residual layer.
        """
        out = self.layers(x)
        return out


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

        return x + self.coral_bias


class OrdinalResLogit(nn.Module):
    def __init__(
        self,
        n_choices,
        alt_spec_features,
        socio_demo_chars,
        n_layers=16,
        batch_size=128,
    ):
        """Initialize the OrdinalResLogit class.

        Args:
        n_vars (int): number of input variables.
        n_choices (int): number of choice alternatives.
        alt_spec_features (list): list of alternative-specific feature names.
        socio_demo_chars (list): list of socio-demographic feature names.
        n_layers (int): number of residual layers.
        batch_size (int): size of each batch.
        """
        super(OrdinalResLogit, self).__init__()

        self.n_choices = n_choices
        self.n_layers = n_layers
        self.batch_size = batch_size

        # define all layers of the model
        self.mnl_sd = MNL(len(socio_demo_chars), n_choices, bias=False)
        self.mnl_as = MNL(len(alt_spec_features), 1, bias=True)
        self.resnet_layer = ResNet(n_choices, n_choices, n_layers=n_layers)
        self.ordinal_layer = Coral_layer(n_choices)

    def forward(self, x, z):
        """return the output of OrdinalResLogit architecture.

        Args:
            x (TensorVariable):  input of OrdinalResLogit architecture that are only in the MNL layers. The alternative-specific features.
            z (TensorVariable):  input of OrdinalResLogit architecture that are only in the ResNet layers. The socio-demographic characteristics.
        """
        z_pre_resnet = self.mnl_sd(z)

        z_post_resnet = self.resnet_layer(z_pre_resnet)

        v = self.mnl_as(x) + z_post_resnet

        #ordinal layer
        logits = self.ordinal_layer(v)

        return logits

        
