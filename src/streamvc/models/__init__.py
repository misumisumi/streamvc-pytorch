import hydra
import torch
import torch.nn as nn
import streamvc.models.encoder
import streamvc.models.decoder

from streamvc.utils import fixed


class StreamVC(nn.Module):
    def __init__(self, config):
        super(StreamVC, self).__init__()
        self.encoder = hydra.utils.instantiate(config.encoder)
        self.decoder = hydra.utils.instantiate(config.decoder)
        if config.spk_encoder is not None:
            self.spk_encoder = hydra.utils.instantiate(config.spk_encoder)
        else:
            self.spk_encoder = None

    def forward_encoder(self, x):
        h = self.encoder(x)

        return h

    def forward_decoder(self, h, s):
        y = self.decoder(h, s)

        return y

    def forward(self, wav, c=None, s=None):
        h = self.forward_encoder(wav)
        h = h.detach()
        if c is not None:
            h = torch.cat([h, c], dim=1)
        if self.spk_encoder is not None and s is not None:
            s = self.spk_encoder(s)
        y = self.forward_decoder(h, s)

        return y
