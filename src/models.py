"""Multi-task UNet architecture and GradNorm loss balancer.

Provides:
    - MultiTaskUNet: Hard parameter-sharing UNet with shared encoder,
      separate segmentation head (via SMP decoder) and classification head.
    - GradNormBalancer: Dynamic multi-task loss weighting following
      Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
      Balancing in Multi-task Learning", ICML 2018.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class GradNormBalancer(nn.Module):
    """GradNorm dynamic loss balancer for two tasks.

    Maintains learnable log-weights for segmentation and classification losses.
    After each batch, gradient magnitudes on shared parameters are balanced
    so that neither task dominates training.

    Args:
        init_seg: Initial segmentation loss weight.
        init_cls: Initial classification loss weight.
        alpha: GradNorm asymmetry exponent (default 1.5).
    """

    def __init__(self, init_seg: float, init_cls: float, alpha: float):
        super().__init__()
        init = torch.tensor([float(init_seg), float(init_cls)], dtype=torch.float32)
        self.log_weights = nn.Parameter(torch.log(init.clamp(min=1e-4)))
        self.alpha = float(alpha)
        self.register_buffer("initial_losses", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("has_initial_losses", torch.tensor(False))

    def weights(self) -> torch.Tensor:
        """Return current (positive) loss weights."""
        return torch.exp(self.log_weights)

    def normalize_(self) -> None:
        """Normalize weights so they sum to 2."""
        with torch.no_grad():
            w = self.weights().clamp(min=1e-4)
            w = w * (2.0 / w.sum().clamp(min=1e-4))
            self.log_weights.copy_(w.log())

    def set_initial_losses(self, seg_loss: torch.Tensor, cls_loss: torch.Tensor) -> None:
        """Record initial per-task losses for relative-rate computation."""
        with torch.no_grad():
            self.initial_losses.copy_(
                torch.tensor([float(seg_loss), float(cls_loss)], device=self.initial_losses.device)
            )
            self.has_initial_losses.fill_(True)


class MultiTaskUNet(nn.Module):
    """Multi-task U-Net with hard parameter sharing.

    Uses a shared SMP Unet encoder for feature extraction. The decoder +
    segmentation head produces pixel-wise masks, while a lightweight
    classification head (global-pool → MLP) produces class logits.

    Args:
        encoder_name: SMP encoder backbone (e.g., 'vgg16', 'mobilenet_v2').
        num_classes: Number of classification categories.
        seg_classes: Number of segmentation classes (1 = binary sigmoid).
        skip_connections: If False, bypass UNet decoder skip connections
            by zeroing intermediate encoder features (ablation study).
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        seg_classes: int,
        skip_connections: bool = True,
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=seg_classes,
            activation=None,
        )
        self._skip_connections = skip_connections
        bottleneck_dim = self.unet.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        bottleneck = features[-1]

        if not self._skip_connections:
            features = [
                torch.zeros_like(f) for f in features[:-1]
            ] + [bottleneck]

        decoder_out = self.unet.decoder(features)
        seg_out = self.unet.segmentation_head(decoder_out)
        cls_out = self.cls_head(bottleneck)
        return seg_out, cls_out