import torch
import torch.nn as nn


class reconstructionKLDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = ""

    def forward(self, inputs, reconstructions, posteriors):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        KLD_loss = 0.5 * torch.sum(posteriors.logvar.exp() - posteriors.logvar - 1 + posteriors.mean.pow(2))

        return rec_loss + KLD_loss




