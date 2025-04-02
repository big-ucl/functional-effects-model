from torch.utils.data import Dataset
import torch

class TasteNetDataset(Dataset):
    def __init__(self, x, y, alt_spec_features, socio_demo_features):
        """
        Parameters:
        ----------
        x : pandas DataFrame
            DataFrame containing the alternative-specific features.
        y : pandas Series
            Series containing the choice outcomes.
        alt_spec_features : list
            List of alternative-specific feature names.
        socio_demo_features : list
            List of socio-demographic feature names.
        """
        self.x = torch.Tensor(x.loc[:, alt_spec_features])
        self.x_names = alt_spec_features
        self.N = len(self.x)
        self.y = torch.Tensor(y)
        
        self.z = torch.Tensor(x.loc[:, socio_demo_features]) # N,D socio-demo variables
        
        _, self.D = self.z.size() # z size = (N,D)

        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        '''
        Get the sample given its idx in the list 
        '''
        return self.x[idx], self.y[idx], self.z[idx]