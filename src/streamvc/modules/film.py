import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    FiLM modulates features using affine transformations conditioned on input.
    Given an input x and conditioning input c, FiLM computes:
        y = gamma * x + beta
    where gamma and beta are derived from c.

    Args:
        in_channels (int): Number of input channels.
        condition_dim (int): Dimension of conditioning input.
    """

    def __init__(self, in_channels: int, condition_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.condition_dim = condition_dim

        # Generate gamma (scale) and beta (bias) from condition
        self.fc = nn.Linear(condition_dim, in_channels * 2)
        # 重みは 0 に初期化
        nn.init.zeros_(self.fc.weight)

        # バイアスの前半（gamma用）を 1.0、後半（beta用）を 0.0 に初期化
        with torch.no_grad():
            self.fc.bias[:in_channels].fill_(1.0)
            self.fc.bias[in_channels:].fill_(0.0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, ...).
            condition (torch.Tensor): Conditioning tensor of shape (B, condition_dim).

        Returns:
            torch.Tensor: Modulated feature tensor of shape (B, C, ...).
        """
        # Generate gamma and beta from condition
        params = self.fc(condition)  # (B, C * 2)
        gamma, beta = torch.chunk(params, 2, dim=-1)  # each (B, C)

        # Reshape for broadcasting
        # Handle different spatial dimensions
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        # Apply FiLM: y = gamma * x + beta
        out = gamma * x + beta

        return out


class FiLMBlock(nn.Module):
    """
    FiLM block combining convolution and FiLM modulation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        condition_dim (int): Dimension of conditioning input.
        kernel_size (int): Kernel size for convolution. Default: 3.
        padding (int): Padding for convolution. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.film = FiLM(out_channels, condition_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution followed by FiLM modulation and activation.

        Args:
            x (torch.Tensor): Input feature tensor.
            condition (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv(x)
        out = self.film(out, condition)
        out = self.relu(out)
        return out
