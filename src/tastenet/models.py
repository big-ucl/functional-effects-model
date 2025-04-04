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
    def __init__(self, args, num_alt_features, num_sd_chars, num_classes):
        super(TasteNet, self).__init__()

        print(type(args.layer_sizes))

        self.params_module = TasteParams(args.layer_sizes, args)
        self.util_module = Utility(args, num_alt_features)
        self.ordinal_module = Coral_layer(num_classes)
        self.args = args
    
    def forward(self, x, z):
        b = self.params_module(z) # taste parameters, (N,1)
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
    def __init__(self, args, num_alt_features):
        super(Utility, self).__init__()
        self.args = args

        self.mnl = nn.Linear(num_alt_features, 1)
        
    def forward(self, x, b):
        '''
        x: attributes of each alternative, 
           including the intercept (N,K)  J alternatives, each have K attributes. 
        b: taste parameters (N, 1): Individual taste parameters.
        '''    
        v = self.mnl(x) + b #(N, J) + (N, 1) = (N, J), ASC is the bias of the linear layer
        
        return v

class TasteParams(nn.Module):
    '''
    Network for tastes
    '''
    def __init__(self, layer_sizes, args):
        """Initialize the TasteParams class.
        Args:
        layer_sizes (list[tuple]): list of layer sizes in a tuple.
        args (argparse.Namespace): command line arguments.
        """
        super(TasteParams, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[0][:-1], layer_sizes[0][1:])):

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