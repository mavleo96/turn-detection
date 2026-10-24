"""Training augmentation with torch-audiomentations, applied per clip (16 kHz) by EoTDataset.

    augment = build_augment()                         # Compose([Gain, AddColoredNoise]) or None
    y = apply_augment(augment, x, seed=(seed, epoch, item))

torch-audiomentations draws from torch's global generator, so `apply_augment` seeds it per clip inside
`torch.random.fork_rng()`: results are reproducible regardless of DataLoader workers, and the caller's
random state is left untouched.
"""

import numpy as np
import torch
from torch_audiomentations import AddColoredNoise, Compose, Gain

from .constants import SAMPLE_RATE

GAIN_DB = 6.0  # uniform random gain in [-gain_db, +gain_db]
NOISE_P = 0.3  # probability of added colored noise ...
SNR_DB = (20.0, 40.0)  # ... at an SNR drawn from this range (vs. the unpadded clip) ...
F_DECAY = (0.0, 2.0)  # ... and a color drawn from this range: 0 white, 1 pink, 2 brown


def build_augment(gain_db: float = GAIN_DB, noise_p: float = NOISE_P, snr_db: tuple = SNR_DB) -> Compose | None:
    """Compose of the enabled transforms (gain_db > 0, noise_p > 0); None when nothing is enabled."""
    transforms = []

    if gain_db > 0:  # Gain rejects an empty range
        transforms.append(
            Gain(
                min_gain_in_db=-gain_db,
                max_gain_in_db=gain_db,
                p=1.0,
                output_type="dict",
            )
        )

    if noise_p > 0:
        transforms.append(
            AddColoredNoise(
                min_snr_in_db=snr_db[0],
                max_snr_in_db=snr_db[1],
                min_f_decay=F_DECAY[0],
                max_f_decay=F_DECAY[1],
                p=noise_p,
                output_type="dict",
            )
        )

    if not transforms:
        return None
    return Compose(transforms, output_type="dict")


def apply_augment(augment: Compose, x: np.ndarray, seed: tuple[int, ...]) -> np.ndarray:
    """Augment one mono float32 waveform, deterministically for `seed`; output clipped to [-1, 1]."""
    clip_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
    samples = torch.from_numpy(x)[None, None]  # (batch 1, channel 1, T), as torch-audiomentations expects

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(clip_seed)
        y = augment(samples, sample_rate=SAMPLE_RATE).samples[0, 0]  # (T,)

    return y.clamp(-1.0, 1.0).numpy()
