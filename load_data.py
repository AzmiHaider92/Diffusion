import numpy as np
import glob

import torch


def load_data_arrays(direc):
    # files are inside direc in format .npy
    array_list = []
    files = glob.glob(glob.escape(direc) + "/*.npy")
    for file in files:
        n = np.load(file)
        n2 = np.reshape(n, (-1, n.shape[-2], n.shape[-1]))
        array_list.append(n2)
    return torch.from_numpy(np.array(array_list))