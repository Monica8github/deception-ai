import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PrimaryCaps(nn.Module):
    """Conv2d → reshape → squash to produce primary capsules."""

    def __init__(
        self,
        in_channels: int = 256,
        num_capsules: int = 8,
        capsule_dim: int = 8,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert convolution outputs into primary capsule tensors."""
        out = self.conv(x)
        B, _, H, W = out.shape
        out = out.view(B, self.num_capsules, self.capsule_dim, H * W)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(B, -1, self.capsule_dim)
        return self.squash(out)

    @staticmethod
    def squash(s: torch.Tensor) -> torch.Tensor:
        """Squash activation for capsule outputs."""
        sq_norm = (s ** 2).sum(dim=-1, keepdim=True)
        return (sq_norm / (1 + sq_norm)) * (s / (sq_norm.sqrt() + 1e-8))


class DigitCaps(nn.Module):
    """Digit capsules with dynamic routing."""

    def __init__(
        self,
        num_capsules: int = 8,
        capsule_dim: int = 16,
        in_capsules: int = None,
        in_dim: int = 8,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(
            torch.randn(1, in_capsules, num_capsules, capsule_dim, in_dim) * 0.01
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply dynamic routing to convert primary capsules into digit capsules."""
        B, N, _ = u.shape
        u_ = u[:, :, None, :, None]
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u_).squeeze(-1)
        return self.routing_by_agreement(u_hat, B, N)

    def routing_by_agreement(
        self,
        u_hat: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Iteratively route votes by agreement between capsules."""
        b = torch.zeros(B, N, self.num_capsules, device=u_hat.device)
        v = None
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = PrimaryCaps.squash(s)
            delta = (u_hat * v.unsqueeze(1)).sum(dim=-1)
            b = b + delta
        return v


class CapsuleNetwork(nn.Module):
    """Visual encoder for video features (6-D vectors)."""

    def __init__(self) -> None:
        super().__init__()
        # Accept (B, 6) audio/visual features and output (B, 256) embedding
        self.fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.EMBED_DIM)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video features into fixed-size embedding.
        
        Args:
            x: (B, 6) or (B, T, 6) video feature vectors
        
        Returns:
            Embedding of shape (B, 256)
        """
        # If input has temporal dimension, average pool it
        if x.dim() == 3:  # (B, T, 6)
            x = x.mean(dim=1)  # (B, 6)
        
        # Pass through MLP
        return self.fc(x)


if __name__ == '__main__':
    net = CapsuleNetwork()
    x = torch.randn(2, 3, 224, 224)
    emb = net(x)
    print(f'CapsuleNetwork output: {emb.shape}')
