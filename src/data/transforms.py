"""Training augmentation with torch-audiomentations, applied per clip (16 kHz) by EoTDataset.

    augment = build_augment(AugmentConfig())          # Compose([Gain, AddColoredNoise]) or None
    y = apply_augment(augment, x, seed=(seed, epoch, item))

torch-audiomentations draws from torch's global generator, so `apply_augment` seeds it per clip inside
`torch.random.fork_rng()`: results are reproducible regardless of DataLoader workers, and the caller's
random state is left untouched.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch_audiomentations import AddColoredNoise, Compose, Gain

from .constants import SAMPLE_RATE


@dataclass
class AugmentConfig:
    """Training augmentation settings (logged with the run)."""

    gain_db: float = 6.0  # uniform random gain in [-gain_db, +gain_db]
    noise_p: float = 0.3  # probability of added white noise ...
    snr_db: tuple[float, float] = (15.0, 40.0)  # ... at an SNR drawn from this range (vs. the whole clip)


def build_augment(cfg: AugmentConfig | None) -> Compose | None:
    """Compose of the enabled transforms (gain_db > 0, noise_p > 0); None when nothing is enabled."""
    if cfg is None:
        return None
    transforms = []
    if cfg.gain_db > 0:  # Gain rejects an empty range
        transforms.append(Gain(min_gain_in_db=-cfg.gain_db, max_gain_in_db=cfg.gain_db, p=1.0, output_type="dict"))
    if cfg.noise_p > 0:
        transforms.append(
            AddColoredNoise(  # f_decay 0: white noise
                min_snr_in_db=cfg.snr_db[0],
                max_snr_in_db=cfg.snr_db[1],
                min_f_decay=0.0,
                max_f_decay=0.0,
                p=cfg.noise_p,
                output_type="dict",
            )
        )
    return Compose(transforms, output_type="dict") if transforms else None


def apply_augment(augment: Compose, x: np.ndarray, seed: tuple[int, ...]) -> np.ndarray:
    """Augment one mono float32 waveform, deterministically for `seed`; output clipped to [-1, 1]."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
        y = augment(torch.from_numpy(x)[None, None], sample_rate=SAMPLE_RATE).samples[0, 0]
    return y.clamp(-1.0, 1.0).numpy()
