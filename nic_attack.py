"""
Compact demo implementation of the proposed two-stage attack for neural image compression.

The file is intentionally self-contained:
- Stage I maximizes global MSE distortion.
- Stage II maximizes block-wise pooled Delta-SSIM degradation.
- Updates use momentum with alternating sign / standardized directions.
- Direct projection enforces the L_inf perturbation budget after each update.
- Optional adaptive frequency masking removes low-frequency components.

Expected codec interface:
    out = codec(x)
    x_hat = out["x_hat"]
where x and x_hat are torch tensors in [0, 1] with shape [N, C, H, W].
"""

from __future__ import annotations

from contextlib import contextmanager
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
    ssim_window: int = 11
    pool_size: int = 32
    block_size: int = 75
    frequency_mask: bool = False
    frequency_threshold: float = 0.075
    use_training_surrogate: bool = True


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [N,3,H,W] in [0,1] to luminance [N,1,H,W]."""
    if x.shape[1] == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Patch SSIM map from ``new_mode_map``, computed without materializing patches."""
    if window < 7 or window % 2 != 1:
        raise ValueError("ssim_window must be odd and at least 7")

    x = rgb_to_y(x)
    y = rgb_to_y(y)
    pad = window // 2
    stats = torch.cat((x, y, x.square(), y.square(), x * y), dim=1)
    stats = F.pad(stats, (pad, pad, pad, pad), mode="reflect")

    coords = torch.arange(7, device=x.device, dtype=x.dtype) - 3
    kernel = torch.exp(-0.5 * (coords / 1.5).square())
    kernel = kernel / kernel.sum()
    stats = F.conv2d(stats, kernel.view(1, 1, 1, 7).expand(5, 1, 1, 7), groups=5)
    stats = F.conv2d(stats, kernel.view(1, 1, 7, 1).expand(5, 1, 7, 1), groups=5)
    mu_x, mu_y, ex2, ey2, exy = stats.chunk(5, dim=1)
    sigma_x2 = ex2 - mu_x.square()
    sigma_y2 = ey2 - mu_y.square()
    sigma_xy = exy - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    similarity = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    # The original loss averages the 7x7 SSIM values within each 11x11 patch.
    return F.avg_pool2d(similarity, kernel_size=window - 6, stride=1)


def block_max_sum(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Sum maxima of non-overlapping blocks, including partial edge blocks."""
    pooled = F.max_pool2d(x, kernel_size=block_size, stride=block_size, ceil_mode=True)
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
    clean_ssim_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage II objective: maximize spatially distributed structural degradation."""
    clean_map = clean_ssim_map
    if clean_map is None:
        clean_map = ssim_map(x, x_hat_clean, window=cfg.ssim_window).detach()
    adv_map = ssim_map(x_adv, x_hat_adv, window=cfg.ssim_window)
    delta = clean_map - adv_map
    # Asymmetric padding keeps the pooled map at H x W for an even pool size.
    left = (cfg.pool_size - 1) // 2
    right = cfg.pool_size // 2
    delta = F.avg_pool2d(F.pad(delta, (left, right, left, right)), cfg.pool_size, stride=1)
    return block_max_sum(delta, block_size=cfg.block_size)


def project_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, cfg: AttackConfig) -> torch.Tensor:
    """Project directly onto [0,1] intersected with the L_inf ball around x_orig."""
    return torch.max(torch.min(x_adv, x_orig + cfg.eps), x_orig - cfg.eps).clamp(0.0, 1.0)


def _frequency_components(delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the spectrum and radial grid once for the adaptive radius search."""
    h, w = delta.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=delta.device),
        torch.arange(w, device=delta.device),
        indexing="ij",
    )
    cy, cx = h // 2, w // 2
    radius_squared = (yy - cy).square() + (xx - cx).square()
    spectrum = torch.fft.fftshift(torch.fft.fft2(delta, dim=(-2, -1)), dim=(-2, -1))
    return spectrum, radius_squared


def _highpass_from_spectrum(
    spectrum: torch.Tensor, radius_squared: torch.Tensor, radius: int
) -> torch.Tensor:
    mask = radius_squared >= radius * radius
    masked = spectrum * mask.view(1, 1, *mask.shape)
    return torch.fft.ifft2(torch.fft.ifftshift(masked, dim=(-2, -1)), dim=(-2, -1)).real


def highpass_frequency_mask(delta: torch.Tensor, radius: int) -> torch.Tensor:
    """Remove low-frequency Fourier coefficients from perturbation delta."""
    if radius <= 0:
        return delta
    return _highpass_from_spectrum(*_frequency_components(delta), radius)


def _direction(momentum: torch.Tensor, iteration: int) -> torch.Tensor:
    """Alternate sign and standardized momentum directions."""
    if iteration % 2 == 1:
        return momentum.sign()
    return (momentum - momentum.mean()) / (momentum.std(unbiased=False) + 1e-8)


@contextmanager
def _surrogate_mode(codec: torch.nn.Module, enabled: bool):
    """Use training-mode quantization for gradients, restoring all module modes."""
    if not enabled:
        yield
        return

    states = [(module, module.training) for module in codec.modules()]
    codec.train()
    # Keep unrelated stochastic layers and running statistics in evaluation mode.
    for module, _ in states:
        if isinstance(
            module,
            (
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.modules.instancenorm._InstanceNorm,
                torch.nn.modules.dropout._DropoutNd,
            ),
        ):
            module.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.training = was_training


def _adaptive_frequency_mask(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    x_hat_clean: torch.Tensor,
    clean_ssim_map: torch.Tensor,
    codec: torch.nn.Module,
    cfg: AttackConfig,
) -> Tuple[torch.Tensor, int]:
    """Choose the largest paper-specified radius that preserves local attack strength."""
    x_hat_adv = codec(x_adv)["x_hat"].clamp(0.0, 1.0)
    strength = local_delta_ssim_loss(
        x_orig, x_adv, x_hat_clean, x_hat_adv, cfg, clean_ssim_map
    ).item()
    if strength <= 0:
        return x_adv, 0

    delta = x_adv - x_orig
    spectrum, radius_squared = _frequency_components(delta)
    for radius in range(100, 0, -10):
        filtered = _highpass_from_spectrum(spectrum, radius_squared, radius)
        candidate = x_orig + filtered
        # Keep the Fourier result intact; reject a radius if it leaves the budget.
        if filtered.abs().max().item() > cfg.eps:
            continue
        if candidate.min().item() < 0 or candidate.max().item() > 1:
            continue
        x_hat_candidate = codec(candidate)["x_hat"].clamp(0.0, 1.0)
        candidate_strength = local_delta_ssim_loss(
            x_orig, candidate, x_hat_clean, x_hat_candidate, cfg, clean_ssim_map
        ).item()
        if candidate_strength >= (1 - cfg.frequency_threshold) * strength:
            return candidate, radius

    return x_adv, 0


def run_attack(
    x: torch.Tensor,
    codec: torch.nn.Module,
    cfg: AttackConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Run the proposed two-stage attack.

    Args:
        x: clean image tensor [1,3,H,W] in [0,1].
        codec: neural image codec returning {"x_hat": ...}; use eval mode so
            clean reconstructions and frequency selection use actual quantization.
        cfg: attack configuration.

    Returns:
        x_adv: adversarial image tensor in [0,1].
        log: dictionary with per-iteration losses and L_inf values.
    """
    if cfg is None:
        cfg = AttackConfig()

    x_orig = x.detach()

    with torch.no_grad():
        x_hat_clean = codec(x_orig)["x_hat"].clamp(0.0, 1.0).detach()

    x_adv = x_orig.clone()
    clean_ssim_map = None
    if cfg.n_local > 0 or cfg.frequency_mask:
        with torch.no_grad():
            clean_ssim_map = ssim_map(x_orig, x_hat_clean, window=cfg.ssim_window).detach()

    momentum = torch.zeros_like(x_adv)
    log: Dict[str, List[float]] = {"loss": [], "linf": [], "frequency_radius": []}

    total_iter = cfg.n_global + cfg.n_local
    for i in range(total_iter):
        if i == cfg.n_global:
            momentum.zero_()
        stage_iteration = i if i < cfg.n_global else i - cfg.n_global
        x_adv = x_adv.detach().requires_grad_(True)

        with _surrogate_mode(codec, cfg.use_training_surrogate):
            out_adv = codec(x_adv)
        x_hat_adv = out_adv["x_hat"].clamp(0.0, 1.0)

        if i < cfg.n_global:
            loss = global_loss(x_adv, x_hat_adv)
            step_size = cfg.lr_global
        else:
            loss = local_delta_ssim_loss(
                x_orig, x_adv, x_hat_clean, x_hat_adv, cfg, clean_ssim_map
            )
            step_size = cfg.lr_local

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        momentum = cfg.momentum * momentum + grad.detach()

        direction = _direction(momentum, stage_iteration)

        with torch.no_grad():
            # Gradient ascent because both objectives are maximized.
            x_adv = x_adv.detach() + step_size * direction
            x_adv = project_linf(x_adv, x_orig, cfg)

            log["loss"].append(float(loss.detach().cpu()))
            log["linf"].append(float((x_adv - x_orig).abs().max().detach().cpu()))

    if cfg.frequency_mask:
        with torch.no_grad():
            x_adv, radius = _adaptive_frequency_mask(
                x_adv, x_orig, x_hat_clean, clean_ssim_map, codec, cfg
            )
            log["frequency_radius"].append(float(radius))

    return x_adv.detach(), log
