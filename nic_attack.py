"""
Compact demo implementation of the proposed two-stage attack for neural image compression.

The file is intentionally self-contained:
- Stage I maximizes global MSE distortion.
- Stage II maximizes block-wise pooled Delta-SSIM degradation.
- Updates use momentum with alternating sign / standardized directions.
- NOpt enforces the L_inf perturbation budget after each update.
- Optional frequency masking removes low-frequency perturbation components.

Expected codec interface:
    out = codec(x)
    x_hat = out["x_hat"]
where x and x_hat are torch tensors in [0, 1] with shape [N, C, H, W].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AttackConfig:
    eps: float = 7 / 255
    lr_global: float = 0.025
    lr_local: float = 0.01
    n_global: int = 15
    n_local: int = 15
    momentum: float = 0.3
    beta: float = 0.001
    nopt_max_iter: int = 550
    init_std: float = 3 / 255
    ssim_window: int = 11
    pool_size: int = 32
    block_size: int = 75
    frequency_mask: bool = False
    frequency_radius: int = 20


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [N,3,H,W] in [0,1] to luminance [N,1,H,W]."""
    if x.shape[1] == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Differentiable SSIM map with a uniform window; returns [N,1,H,W]."""
    x = rgb_to_y(x)
    y = rgb_to_y(y)

    pad = window // 2
    weight = torch.ones(1, 1, window, window, device=x.device, dtype=x.dtype)
    weight = weight / float(window * window)

    mu_x = F.conv2d(x, weight, padding=pad)
    mu_y = F.conv2d(y, weight, padding=pad)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, weight, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, weight, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, weight, padding=pad) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )


def block_max_sum(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Sum of block-wise maxima for a nonnegative map [N,1,H,W].
    Padding is used only to make H,W divisible by block_size.
    """
    n, c, h, w = x.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    pooled = F.max_pool2d(x, kernel_size=block_size, stride=block_size)
    return pooled.sum()


def global_loss(x_adv: torch.Tensor, x_hat_adv: torch.Tensor) -> torch.Tensor:
    """Stage I objective: maximize global reconstruction distortion."""
    return F.mse_loss(x_adv, x_hat_adv, reduction="mean")


def local_delta_ssim_loss(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    x_hat_clean: torch.Tensor,
    x_hat_adv: torch.Tensor,
    cfg: AttackConfig,
) -> torch.Tensor:
    """Stage II objective: maximize spatially distributed structural degradation."""
    clean_map = ssim_map(x, x_hat_clean, window=cfg.ssim_window).detach()
    adv_map = ssim_map(x_adv, x_hat_adv, window=cfg.ssim_window)
    delta = clean_map - adv_map
    delta = F.avg_pool2d(delta, kernel_size=cfg.pool_size, stride=1, padding=cfg.pool_size // 2)
    delta = F.relu(delta)
    return block_max_sum(delta, block_size=cfg.block_size)


def nopt(x_adv: torch.Tensor, x_orig: torch.Tensor, cfg: AttackConfig) -> torch.Tensor:
    """
    Lightweight NOpt correction for the L_inf budget.
    It iteratively moves only violating pixels back toward the feasible region.
    """
    x_adv = x_adv.detach()
    for _ in range(cfg.nopt_max_iter):
        resid = x_adv - x_orig
        violation = resid.abs() - cfg.eps
        if violation.max().item() <= 0:
            break

        mask = violation > 0
        step = cfg.beta * resid.sign() * mask.to(resid.dtype)
        x_adv = (x_adv - step).clamp(0.0, 1.0)

    # Final exact clamp is kept as a safety guard for numerical issues.
    return torch.max(torch.min(x_adv, x_orig + cfg.eps), x_orig - cfg.eps).clamp(0.0, 1.0)


def highpass_frequency_mask(delta: torch.Tensor, radius: int) -> torch.Tensor:
    """Remove low-frequency Fourier coefficients from perturbation delta."""
    if radius <= 0:
        return delta

    n, c, h, w = delta.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=delta.device),
        torch.arange(w, device=delta.device),
        indexing="ij",
    )
    cy, cx = h // 2, w // 2
    mask = (((yy - cy) ** 2 + (xx - cx) ** 2).sqrt() >= radius).to(delta.dtype)
    mask = mask.view(1, 1, h, w)

    spectrum = torch.fft.fftshift(torch.fft.fft2(delta, dim=(-2, -1)), dim=(-2, -1))
    spectrum = spectrum * mask
    filtered = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)).real
    return filtered


def _direction(momentum: torch.Tensor, iteration: int) -> torch.Tensor:
    """Alternate sign and standardized momentum directions."""
    if iteration % 2 == 1:
        direction = momentum.sign()
        return torch.where(direction == 0, torch.ones_like(direction), direction)

    std = momentum.std()
    if std.item() < 1e-8:
        direction = momentum.sign()
        return torch.where(direction == 0, torch.ones_like(direction), direction)

    return (momentum - momentum.mean()) / (std + 1e-8)


def run_attack(
    x: torch.Tensor,
    codec: torch.nn.Module,
    cfg: AttackConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Run the proposed two-stage attack.

    Args:
        x: clean image tensor [1,3,H,W] in [0,1].
        codec: neural image codec returning {"x_hat": ...}.
        cfg: attack configuration.

    Returns:
        x_adv: adversarial image tensor in [0,1].
        log: dictionary with per-iteration losses and L_inf values.
    """
    if cfg is None:
        cfg = AttackConfig()

    device = x.device
    x_orig = x.detach()

    with torch.no_grad():
        x_hat_clean = codec(x_orig)["x_hat"].clamp(0.0, 1.0).detach()

    x_adv = x_orig + torch.randn_like(x_orig) * cfg.init_std
    x_adv = torch.max(torch.min(x_adv, x_orig + cfg.eps), x_orig - cfg.eps).clamp(0.0, 1.0)

    momentum = torch.zeros_like(x_adv, device=device)
    log: Dict[str, List[float]] = {"loss": [], "linf": []}

    total_iter = cfg.n_global + cfg.n_local
    for i in range(total_iter):
        x_adv = x_adv.detach().requires_grad_(True)

        out_adv = codec(x_adv)
        x_hat_adv = out_adv["x_hat"].clamp(0.0, 1.0)

        if i < cfg.n_global:
            loss = global_loss(x_adv, x_hat_adv)
            step_size = cfg.lr_global
        else:
            loss = local_delta_ssim_loss(x_orig, x_adv, x_hat_clean, x_hat_adv, cfg)
            step_size = cfg.lr_local

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        momentum = cfg.momentum * momentum + grad.detach()

        direction = _direction(momentum, i)

        with torch.no_grad():
            # Gradient ascent because both objectives are maximized.
            x_adv = x_adv.detach() + step_size * direction
            x_adv = x_adv.clamp(0.0, 1.0)
            x_adv = nopt(x_adv, x_orig, cfg)

            log["loss"].append(float(loss.detach().cpu()))
            log["linf"].append(float((x_adv - x_orig).abs().max().detach().cpu()))

    if cfg.frequency_mask:
        with torch.no_grad():
            delta = highpass_frequency_mask(x_adv - x_orig, cfg.frequency_radius)
            x_adv = (x_orig + delta).clamp(0.0, 1.0)
            x_adv = torch.max(torch.min(x_adv, x_orig + cfg.eps), x_orig - cfg.eps).clamp(0.0, 1.0)

    return x_adv.detach(), log
