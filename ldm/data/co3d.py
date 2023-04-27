import os.path
from load_data import load_data_arrays
from torch.utils.data import Dataset, Subset
import numpy as np
import torch


class co3D_dataset(Dataset):
    # This loads the co3d_data and converts it, make co3d_data rdy
    def __init__(self, data_dir):
        # load co3d_data
        self.directory = data_dir
        self.data = load_data_arrays(data_dir)
        # conver to torch dtypes
        self.dataset = torch.tensor(self.data).float()

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx]