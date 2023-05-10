import torch
import torch.nn as nn


class reconstructionKLDLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)


    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # reconstruction loss
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # negative log likelihood loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # KL
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/logvar".format(split): self.logvar.detach(),
               "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/nll_loss".format(split): nll_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        #KLD_loss = 0.5 * torch.sum(posteriors.logvar.exp() - posteriors.logvar - 1 + posteriors.mean.pow(2))

        return loss, log




