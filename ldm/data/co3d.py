import os.path
from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import glob


def load_data_arrays(direc):
    # files are inside direc in format .npy
    array_list = []
    files = glob.glob(glob.escape(direc) + "/*.npy")
    for file in files:
        n = np.load(file)
        n2 = np.reshape(n, (-1, n.shape[-2], n.shape[-1]))
        array_list.append(n2)
    return torch.from_numpy(np.array(array_list))


class co3D_dataset(Dataset):
    # This loads the co3d_data and converts it, make co3d_data rdy
    def __init__(self, data_dir):
        # load co3d_data
        self.directory = data_dir
        self.data_paths = glob.glob(data_dir + "/*.npy")
        # conver to torch dtypes

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.data_paths)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        item = np.load(self.data_paths[idx])
        item = np.reshape(item, (-1, item.shape[-2], item.shape[-1]))
        item = torch.Tensor(item)
        return item
