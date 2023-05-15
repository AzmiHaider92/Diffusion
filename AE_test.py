import os
import numpy as np
import torch
torch.cuda.empty_cache()
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

if __name__ == "__main__":
    # load model
    checkpoint_folder = r'ldm\logs\2023-05-11T19-50-50_co3d_VQautoencoder'
    configs = os.path.join(checkpoint_folder, '2023-05-11T19-50-50-project.yaml')
    model_ckpt = os.path.join(checkpoint_folder, 'last.ckpt')
    configs = OmegaConf.load(configs)
    model = instantiate_from_config(configs.model)
    model.load_state_dict(torch.load(model_ckpt)["state_dict"])
    model.eval()

    # predict
    original = np.load(r'C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\AE\108_12867_22800\108_12867_22800.npy')
    item = np.reshape(original, (-1, original.shape[-2], original.shape[-1]))
    item = torch.Tensor(item)
    item = item[None,:]
    reconstructed = model(item)
    rec_item = torch.squeeze(reconstructed[0])
    rec_item = np.reshape(rec_item.detach().numpy(), original.shape)
    np.save(r'C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\AE\108_12867_22800\108_12867_22800_rec.npy', rec_item)




