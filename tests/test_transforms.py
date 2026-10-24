import numpy as np
import torch
from torch_audiomentations import AddColoredNoise, Compose, Gain

from turn_detection.data.transforms import apply_augment, build_augment


def speech(n: int = 16000) -> np.ndarray:
    return (0.1 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)


def test_build_augment():
    augment = build_augment()
    assert isinstance(augment, Compose)
    gain, noise = augment.transforms
    assert isinstance(gain, Gain) and (gain.min_gain_in_db, gain.max_gain_in_db, gain.p) == (-6.0, 6.0, 1.0)
    assert isinstance(noise, AddColoredNoise) and noise.p == 0.3
    assert (noise.min_snr_in_db, noise.max_snr_in_db, noise.min_f_decay, noise.max_f_decay) == (20.0, 40.0, 0.0, 2.0)
    assert build_augment(gain_db=0.0, noise_p=0.0) is None
    (only_noise,) = build_augment(gain_db=0.0).transforms
    assert isinstance(only_noise, AddColoredNoise)


def test_gain_only_stays_in_range():
    x = speech()
    augment = build_augment(noise_p=0.0)
    for seed in range(20):
        ratio = np.abs(apply_augment(augment, x, (seed,))).max() / np.abs(x).max()
        assert 10 ** (-6 / 20) - 1e-4 <= ratio <= 10 ** (6 / 20) + 1e-4
    loud = np.full(1000, 0.9, dtype=np.float32)
    assert np.abs(apply_augment(build_augment(gain_db=20.0, noise_p=0.0), loud, (1,))).max() <= 1.0


def test_noise_snr():
    x = speech()
    augment = build_augment(gain_db=0.0, noise_p=1.0, snr_db=(20.0, 20.0))
    noise = apply_augment(augment, x, (0,)) - x
    snr = 20 * np.log10(np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(noise**2)))
    assert abs(snr - 20) < 0.5


def test_deterministic_per_seed_and_leaves_global_rng_alone():
    x = speech()
    augment = build_augment(noise_p=1.0)
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    a = apply_augment(augment, x, (1, 2, 3))
    assert torch.equal(torch.rand(1), expected_next)  # the caller's random stream is untouched
    b = apply_augment(augment, x, (1, 2, 3))
    assert a.shape == x.shape and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, apply_augment(augment, x, (1, 2, 4)))
