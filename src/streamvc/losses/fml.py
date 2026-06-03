import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchLoss(nn.Module):
    # Feature matching loss module.

    def __init__(self, average_by_layers=False, average_by_fmap_real=False):
        """Initialize FeatureMatchLoss module."""
        super(FeatureMatchLoss, self).__init__()
        self.average_by_layers = average_by_layers
        self.average_by_fmap_real = average_by_fmap_real

    def forward(self, fmaps_fake, fmaps_real):
        """Calculate forward propagation.

        Args:
            fmaps_fake (list): List of discriminator outputs
                calculated from generator outputs.
            fmaps_real (list): List of discriminator outputs
                calculated from groundtruth.

        Returns:
            Tensor: Feature matching loss value.

        """
        fm_loss = 0.0
        for feat_fake, feat_real in zip(fmaps_fake, fmaps_real):
            loss = F.l1_loss(feat_fake, feat_real.detach())
            if self.average_by_fmap_real:
                # Feature Matching Loss of EnCodec
                _mean = torch.clamp(torch.mean(torch.abs(feat_real.detach())), min=1e-8)
                loss = loss / _mean
            fm_loss = fm_loss + loss

        if self.average_by_layers:
            fm_loss = fm_loss / len(fmaps_fake)

        return fm_loss
