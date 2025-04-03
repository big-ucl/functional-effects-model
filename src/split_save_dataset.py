from utils import split_dataset
from constants import PATH_TO_DATA
import pandas as pd

def split_save_dataset():
    """
    Train the specified model.
    """
    # load the data
    data = pd.read_csv(PATH_TO_DATA)

    features = [
        col
        for col in data.columns
        if col not in ["mergeid", "hhid", "coupleid", "depression_scale"]
    ]
    target = "depression_scale"

    # split data
    split_dataset(
        data,
        target,
        features,
        train_size=0.64,
        val_size=0.16,
        groups=data["hhid"],
        random_state=42,
        save_path= PATH_TO_DATA[:-4] + "_",
    )

if __name__ == "__main__":
    split_save_dataset()
