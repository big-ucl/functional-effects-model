from torch import nn
import torch

class DNN(nn.Module):

    def __init__(self, layers: list[int], activation: str = 'relu', dropout: float = 0.0, batch_norm: bool = False, input_dim: int = None, output_dim: int = None, device=None):
        """
        Parameters
        ----------
        layers: list[int]
            The number of units in each layer.
        activation: str
            The activation function to use.
        dropout: float
            The dropout rate.
        batch_norm: bool
            Whether to use batch normalisation.
        input_dim: int
            The input dimension.
        output_dim: int
            The output dimension.
        """
        super(DNN, self).__init__()
        self.layers = layers
        self.layers.insert(0, input_dim)
        self.layers.append(output_dim)
        self.activation = get_act(activation)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.dnn = self._build_model()
        self.device = device

    def _build_model(self) -> nn.Sequential:

        seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            seq.add_module(
                name=f"L{i+1}", module=nn.Linear(in_size, out_size, bias=True)
            )
            if i < len(self.layers) - 2:
                seq.add_module(name=f"A{i+1}", module=self.activation)
                if self.dropout > 0:
                    seq.add_module(name=f"D{i+1}", module=nn.Dropout(self.dropout))
                if self.batch_norm:
                    seq.add_module(
                        name=f"BN{i+1}", module=nn.BatchNorm1d(out_size)
                    )

        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, input_dim)
        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, output_dim)
        """
        return self.dnn(x)  # (N,K)
    
    def l2_norm(self):
        """
        L2 norm, not including bias
        """
        norm = torch.zeros(1).to(device=torch.device(self.device))
        for i, params in enumerate(self.dnn.parameters()):
            if i % 2 == 1:
                # skip bias
                continue
            norm += (params**2).sum()

        return norm

    def l1_norm(self):
        """
        L1 norm, not including bias
        """
        norm = torch.zeros(1).to(device=torch.device(self.device))
        for i, params in enumerate(self.dnn.parameters()):
            if i % 2 == 1:
                # skip bias
                continue
            norm += torch.abs(params).sum()

        return norm
    

def get_act(nl_func):
    if nl_func == "tanh":
        return nn.Tanh()
    elif nl_func == "relu":
        return nn.ReLU()
    elif nl_func == "sigmoid":
        return nn.Sigmoid()
    else:
        return None
    