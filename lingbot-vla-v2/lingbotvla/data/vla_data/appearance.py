"""Global appearance randomization of PatchWAM's RoboTwin clean-to-randomized (C2R) recipe.

Ported from PatchWAM ``src/patchwam/data/appearance.py`` (``AppearanceRandomizer``, recipe C2R-DR4, the
``video_augmentation`` of ``configs/robotwin_c2r.yaml``); random numbers come from a generator passed in, so
dataloader workers on different ranks do not repeat each other. Original notice:

    SPDX-License-Identifier: MIT
    Copyright (c) 2026 Yuyang "Alice.L"

Geometry-preserving: photometric changes (brightness / contrast / saturation / hue in random order, gamma,
exposure, colour temperature, blur, Gaussian noise), then per-channel statistics moved toward random targets
(an AdaIN-like style change) and a smooth random modulation of the Fourier amplitude. One draw decides whether a
sample is changed; each camera gets its own parameters, shared by all of that camera's frames.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as vision

DEFAULTS = {
    "prob": 0.0,  # PatchWAM C2R: 0.8
    "photometric": {
        "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1,
        "gamma": [0.7, 1.4], "exposure": [-0.5, 0.5], "color_temperature": 0.15,
        "gaussian_noise_std": 0.03, "blur_sigma": [0.0, 1.5], "blur_prob": 0.3,
    },
    "style": {"enabled": True, "p": 0.5, "target_mean": [0.2, 0.8], "target_std": [0.05, 0.4],
              "strength": [0.4, 1.0], "eps": 1e-5},
    "fourier": {"enabled": True, "p": 0.5, "amp_jitter": [0.5, 1.5], "envelope_strength": [0.0, 0.6],
                "envelope_grid": 8},
}


def _uniform(gen, low, high):
    return float(torch.empty(()).uniform_(float(low), float(high), generator=gen))


def _lighting(frames, cfg, gen):
    operations = []
    for key, function in (("brightness", vision.adjust_brightness), ("contrast", vision.adjust_contrast),
                          ("saturation", vision.adjust_saturation), ("hue", vision.adjust_hue)):
        amount = float(cfg[key])
        if amount > 0:
            bounds = (-min(amount, 0.5), min(amount, 0.5)) if key == "hue" else (max(0, 1 - amount), 1 + amount)
            operations.append((function, _uniform(gen, *bounds)))
    for position in torch.randperm(len(operations), generator=gen).tolist():
        function, factor = operations[position]
        frames = function(frames, factor)
    if cfg["gamma"]:
        frames = vision.adjust_gamma(frames.clamp(0, 1), _uniform(gen, *cfg["gamma"]))
    if cfg["exposure"]:
        frames = frames * 2 ** _uniform(gen, *cfg["exposure"])
    temperature = float(cfg["color_temperature"])
    if temperature > 0:
        gains = torch.tensor([1 + _uniform(gen, -temperature, temperature),
                              1 + _uniform(gen, -0.5 * temperature, 0.5 * temperature),
                              1 + _uniform(gen, -temperature, temperature)], dtype=frames.dtype).view(1, 3, 1, 1)
        frames = frames * gains
    frames = frames.clamp(0, 1)
    if cfg["blur_sigma"] and _uniform(gen, 0, 1) < float(cfg["blur_prob"]):
        sigma = _uniform(gen, max(1e-3, cfg["blur_sigma"][0]), cfg["blur_sigma"][1])
        if sigma > 1e-2:
            kernel = int(2 * round(3 * sigma) + 1)
            frames = vision.gaussian_blur(frames, [kernel, kernel], [sigma, sigma])
    if float(cfg["gaussian_noise_std"]) > 0:
        noise = torch.randn((1, *frames.shape[1:]), generator=gen, dtype=frames.dtype)
        frames = frames + noise * float(cfg["gaussian_noise_std"])
    return frames.clamp(0, 1)


def _channel_statistics(frames, cfg, gen):
    if not cfg["enabled"] or _uniform(gen, 0, 1) >= float(cfg["p"]):
        return frames
    channels = frames.shape[1]
    flattened = frames.permute(1, 0, 2, 3).reshape(channels, -1)
    mean = flattened.mean(1).view(1, channels, 1, 1)
    std = flattened.std(1).view(1, channels, 1, 1).clamp_min(float(cfg["eps"]))
    target_mean = torch.empty(channels, dtype=frames.dtype).uniform_(*cfg["target_mean"], generator=gen).view(1, channels, 1, 1)
    target_std = torch.empty(channels, dtype=frames.dtype).uniform_(*cfg["target_std"], generator=gen).view(1, channels, 1, 1)
    strength = _uniform(gen, *cfg["strength"])
    effective_mean = mean + strength * (target_mean - mean)
    effective_std = std + strength * (target_std - std)
    return ((frames - mean) / std * effective_std + effective_mean).clamp(0, 1)


def _spectrum(frames, cfg, gen):
    if not cfg["enabled"] or _uniform(gen, 0, 1) >= float(cfg["p"]):
        return frames
    spectrum = torch.fft.fft2(frames.float(), dim=(-2, -1))
    amplitude, phase = spectrum.abs(), spectrum.angle()
    grid = int(cfg["envelope_grid"])
    jitter = torch.empty(1, 1, grid, grid).uniform_(*cfg["amp_jitter"], generator=gen)
    jitter = F.interpolate(jitter, size=frames.shape[-2:], mode="bilinear", align_corners=False)
    strength = _uniform(gen, *cfg["envelope_strength"])
    envelope = torch.empty(1, 1, grid, grid).uniform_(0.5, 1.5, generator=gen)
    envelope = F.interpolate(envelope, size=frames.shape[-2:], mode="bilinear", align_corners=False)
    modulation = jitter * (1 - strength) + envelope * strength
    modified = torch.polar(amplitude * modulation.clamp_min(0), phase)
    return torch.fft.ifft2(modified, dim=(-2, -1)).real.to(frames.dtype).clamp(0, 1)


def randomize_view(frames, cfg, gen):
    """frames: float TCHW RGB in [0, 1], the frames of one camera (current, future) -> same shape."""
    return _spectrum(_channel_statistics(_lighting(frames, cfg["photometric"], gen), cfg["style"], gen),
                     cfg["fourier"], gen)
