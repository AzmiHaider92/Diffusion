import torch
from diffusion_model import create_diffusion_model
import numpy as np
from opt import config_parser
from load_data import load_data_arrays
import sys
from tqdm import tqdm


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()

    training_images = load_data_arrays(args.images_dir)  # images are normalized from 0 to 1
    diffusion = create_diffusion_model(args.images_dim, args.images_channel)
    pbar = tqdm(range(args.num_epochs), miniters=1, file=sys.stdout)
    for epoch in pbar:
        loss = diffusion(training_images)
        loss.backward()

        l = loss.detach().item()
        pbar.set_description(
            f'Iteration {epoch:05d}:'
            + f' mse = {l:.6f}')

    sampled_planes = diffusion.sample(batch_size=1)

