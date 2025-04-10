import torch
import torch.nn as nn
import torch.nn.functional as F

def get_act(nl_func):
    if nl_func=="tanh":
        return nn.Tanh()
    elif nl_func == "relu":
        return nn.ReLU()
    elif nl_func == "sigmoid":
        return nn.Sigmoid()
    else:
        return None
    
class TasteNet(nn.Module):
    """TasteNet-MNL model for Swissmetro"""
    def __init__(self, args, num_alt_features, num_sd_chars, num_classes, num_latent_vals=None):
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
            self.params_module = TasteParams(args.layer_sizes, args, self.func_intercept, self.func_params, num_alt_features, num_latent_vals, num_sd_chars)
        self.util_module = Utility(args, num_alt_features, num_latent_vals, self.func_intercept, self.func_params)
        self.ordinal_module = Coral_layer(num_classes)
        self.args = args
    
    def forward(self, x, z=None):
        if self.func_intercept or self.func_params:
            b = self.params_module(z) # taste parameters, (N,1)
        else:
            b = None
        v = self.util_module(x,b) #no softmax here 

        logits = self.ordinal_module(v) # (N, J-1)
        
        return logits  
    
    def L2Norm(self):
        '''
        L2 norm, not including bias
        '''
        norm = torch.zeros(1)
        for params in self.parameters():
            norm += (params**2).sum()
        return norm            

    def L1Norm(self):
        '''
        L1 norm, not including bias
        '''
        norm = torch.zeros(1)
        for params in self.parameters():
            norm += (torch.abs(params).sum())
        return norm

class Utility(nn.Module):
    def __init__(self, args, num_alt_features, num_classes, func_intercept=True, func_params=True):
        super(Utility, self).__init__()
        self.args = args

        self.func_intercept = func_intercept
        self.func_params = func_params

        self.num_classes = num_classes

        if not self.func_params:
            self.mnl = nn.Linear(num_alt_features, num_classes)
        
        if not self.func_intercept:
            self.intercept = nn.Parameter(torch.zeros(num_classes)) # (1, J)

        if self.func_intercept and self.func_params:
            self.ones = torch.ones(args.batch_size, 1).to(args.device) # (N, J)
        
    def forward(self, x, b=None):
        '''
        x: attributes of each alternative, 
           including the intercept (N,K)  J alternatives, each have K attributes. 
        b: taste parameters (N, 1): Individual taste parameters.
        ''' 
        if not self.func_params and not self.func_intercept:
            v = self.mnl(x) + self.intercept
        elif self.func_params and not self.func_intercept:
            v = torch.matmul(x, b.view(-1, self.num_classes)) + self.func_intercept.view(1, self.num_classes) # (N, K) @ (K, J) = (N, J)
        elif not self.func_params and self.func_intercept:
            v = self.mnl(x) + b
        else:
            x = torch.cat((x, self.ones), dim=1) # (N, K+1)
            v = torch.matmul(x, b.view(-1, self.num_classes + 1))
        
        return v

class TasteParams(nn.Module):
    '''
    Network for tastes
    '''
    def __init__(self, layer_sizes, args, num_alt_features, num_classes, num_sd_chars, func_intercept=True, func_params=True):
        """Initialize the TasteParams class.
        Args:
        layer_sizes (list[tuple]): list of layer sizes in a tuple.
        args (argparse.Namespace): command line arguments.
        func_intercept (bool): whether to include functional intercepts.
        func_params (bool): whether to include functional taste parameters.
        """
        if not func_intercept and not func_params:
            raise ValueError("At least one of func_intercept or func_params must be True.")
        all_layers = [l for l in layer_sizes]
        all_layers.insert(0, num_sd_chars)
        all_layers.append(num_alt_features * num_classes * func_params + num_classes * func_intercept) 

        super(TasteParams, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.seq.add_module(name=f"L{i+1}", module=nn.Linear(in_size, out_size, bias=True))
            if i < len(layer_sizes) - 2:
                self.seq.add_module(name=f"A{i+1}", module=get_act(args.act_func))
                if args.dropout > 0:
                    self.seq.add_module(name=f"D{i+1}", module=nn.Dropout(args.dropout))
                if args.batch_norm:
                    self.seq.add_module(name=f"BN{i+1}", module=nn.BatchNorm1d(out_size))
        self.args = args
        
    def forward(self,z):
        '''
        Parameters:
            z: (N,D) # batch size, input dimension
        Returns:
            V: (N,1) # taste parameters 
        '''
        return self.seq(z) # (N,K) 


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