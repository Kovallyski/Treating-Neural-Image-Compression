"""
Demo run of the proposed attack on one image and one Cheng2020 codec.

Example:
    python demo_cheng_attack.py --image kodim01.png --quality 6 --device cuda

Outputs are saved to ./demo_outputs by default:
    original.png
    adversarial_input.png
    clean_reconstruction.png
    attacked_reconstruction.png
    perturbation_x10.png
    losses.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from compressai.zoo import cheng2020_attn

from nic_attack import AttackConfig, run_attack


def load_image(path: str, multiple: int = 64) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    h, w = x.shape[-2:]
    h2 = (h // multiple) * multiple
    w2 = (w // multiple) * multiple
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Image is too small for multiple={multiple}: {path}")

    if h2 != h or w2 != w:
        x = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)

    return x.clamp(0.0, 1.0)


def save_tensor_image(x: torch.Tensor, path: Path) -> None:
    x = x.detach().clamp(0.0, 1.0).cpu().squeeze(0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y, reduction="mean").item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def build_codec(device: str, quality: int):
    codec = cheng2020_attn(quality=quality, pretrained=True).eval().to(device)
    for param in codec.parameters():
        param.requires_grad_(False)
    return codec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to one RGB image.")
    parser.add_argument("--out_dir", default="demo_outputs", help="Directory for demo outputs.")
    parser.add_argument("--quality", type=int, default=6, help="Cheng2020 quality level.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eps", type=float, default=7 / 255)
    parser.add_argument("--n_global", type=int, default=15)
    parser.add_argument("--n_local", type=int, default=15)
    parser.add_argument("--lr_global", type=float, default=0.025)
    parser.add_argument("--lr_local", type=float, default=0.01)
    parser.add_argument("--frequency_mask", action="store_true")
    parser.add_argument("--frequency_threshold", type=float, default=0.075)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = load_image(args.image).to(device)
    codec = build_codec(str(device), args.quality)

    cfg = AttackConfig(
        eps=args.eps,
        lr_global=args.lr_global,
        lr_local=args.lr_local,
        n_global=args.n_global,
        n_local=args.n_local,
        frequency_mask=args.frequency_mask,
        frequency_threshold=args.frequency_threshold,
    )

    with torch.no_grad():
        x_hat_clean = codec(x)["x_hat"].clamp(0.0, 1.0)

    start = time.time()
    x_adv, log = run_attack(x, codec, cfg)
    elapsed = time.time() - start

    with torch.no_grad():
        x_hat_attacked = codec(x_adv)["x_hat"].clamp(0.0, 1.0)

    linf = float((x_adv - x).abs().max().detach().cpu())
    input_psnr = psnr(x, x_adv)
    clean_psnr = psnr(x, x_hat_clean)
    attacked_psnr = psnr(x, x_hat_attacked)

    save_tensor_image(x, out_dir / "original.png")
    save_tensor_image(x_adv, out_dir / "adversarial_input.png")
    save_tensor_image(x_hat_clean, out_dir / "clean_reconstruction.png")
    save_tensor_image(x_hat_attacked, out_dir / "attacked_reconstruction.png")

    # Visualization only: amplify perturbation around neutral gray.
    perturbation_vis = ((x_adv - x) * 10.0 + 0.5).clamp(0.0, 1.0)
    save_tensor_image(perturbation_vis, out_dir / "perturbation_x10.png")

    with open(out_dir / "losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "loss", "linf"])
        for i, (loss, cur_linf) in enumerate(zip(log["loss"], log["linf"])):
            writer.writerow([i, loss, cur_linf])

    print("Demo finished")
    print(f"Image: {args.image}")
    print(f"Codec: cheng2020_attn, quality={args.quality}")
    print(f"Runtime: {elapsed:.2f} sec")
    print(f"L_inf perturbation: {linf:.6f}  (eps={args.eps:.6f})")
    print(f"Input PSNR(original, adversarial): {input_psnr:.2f} dB")
    print(f"Clean reconstruction PSNR: {clean_psnr:.2f} dB")
    print(f"Attacked reconstruction PSNR: {attacked_psnr:.2f} dB")
    print(f"Delta PSNR: {clean_psnr - attacked_psnr:.2f} dB")
    if log["frequency_radius"]:
        print(f"Selected frequency radius: {int(log['frequency_radius'][0])}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
