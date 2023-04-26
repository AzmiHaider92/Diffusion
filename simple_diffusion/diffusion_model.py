import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


def create_diffusion_model(image_dim, image_channels):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=image_channels
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=image_dim,
        timesteps=1000,   # number of steps
        loss_type='l1'    # L1 or L2
    )

    return diffusion

