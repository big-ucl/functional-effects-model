from torch.utils.data import Dataset

class ResLogitDataset(Dataset):
    """Custom dataset for ResLogit model."""

    def __init__(self, x, y):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the dataset.
        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        """
        Get the sample given its idx in the list 
        """
        x = self.x.iloc[idx]
        y = self.y.iloc[idx]
        return x, y