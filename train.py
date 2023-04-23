import torch
from diffusion_model import create_diffusion_model
import numpy as np
from opt import config_parser
from load_data import load_data_arrays
import sys
from tqdm import tqdm


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()

    training_images = load_data_arrays(args.images_dir)  # images are normalized from 0 to 1
    diffusion = create_diffusion_model(args.images_dim, args.images_channel)
    pbar = tqdm(range(args.num_epochs), miniters=1, file=sys.stdout)

    trainingSampler = SimpleSampler(training_images.shape[0], args.batch_size)
    for epoch in pbar:
        # batch
        ids = trainingSampler.nextids()
        batch = training_images[ids]

        # model update
        loss = diffusion(batch)
        loss.backward()

        # result
        l = loss.detach().item()
        pbar.set_description(
            f'Iteration {epoch:05d}:'
            + f' mse = {l:.6f}')

    # save model
    torch.save(diffusion.state_dict(), './checkpoints/diffusion.pt')
    #sampled_planes = diffusion.sample(batch_size=1)

