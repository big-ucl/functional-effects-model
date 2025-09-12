from torch.utils.data import Dataset
import torch

class DNNDataset(Dataset):
    def __init__(self, x, y, features):
        """
        Parameters:
        ----------
        x : pandas DataFrame
            DataFrame containing the alternative-specific features.
        y : pandas Series
            Series containing the choice outcomes.
        """
        self.x = torch.from_numpy(x[features].values).to(dtype=torch.float32)
        self.N = len(self.x)
        self.y = torch.from_numpy(y.values)

        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        '''
        Get the sample given its idx in the list 
        '''
        return self.x[idx], self.y[idx]