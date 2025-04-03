from torch.utils.data import Dataset
import torch

class ResLogitDataset(Dataset):
    """Custom dataset for ResLogit model."""

    def __init__(self, x, y, alt_spec_features, socio_demo_features):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the dataset.
        """
        self.x = torch.from_numpy(x.loc[:, alt_spec_features].values).to(dtype=torch.float32) # N,A alternative-specific variables
        self.x_names = alt_spec_features
        self.N = len(self.x)
        self.y = torch.from_numpy(y.values)

        self.z = torch.from_numpy(x.loc[:, socio_demo_features].values).to(dtype=torch.float32) # N,D socio-demo variables

    def __len__(self):
        return self.N 
    
    def __getitem__(self, idx):
        """
        Get the sample given its idx in the list 
        """
        return self.x[idx], self.y[idx], self.z[idx]