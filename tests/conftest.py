import pytest

import explore_dataset
import samples
from turn_detection.data import SMART_TURN_TEST, SMART_TURN_TRAIN, TTD, dataset

from .helpers import fake_hf_split, fake_ttd_split


@pytest.fixture
def hub(monkeypatch):
    """Tiny in-memory splits of both datasets standing in for the cached Hugging Face datasets (no network)."""
    fakes = {
        "smart-turn": {"train": fake_hf_split(80, "train"), "test": fake_hf_split(30, "test", seed=1)},
        "ttd": {
            "train": fake_ttd_split(40, "train"),
            "validation": fake_ttd_split(16, "validation", seed=1),
            "test": fake_ttd_split(20, "test", seed=2),
        },
    }
    by_repo = {
        (SMART_TURN_TRAIN, "train"): fakes["smart-turn"]["train"],
        (SMART_TURN_TEST, "train"): fakes["smart-turn"]["test"],
    }
    by_repo |= {(TTD, split): ds for split, ds in fakes["ttd"].items()}
    # Note: every module that calls load_dataset gets the fakes
    for module in (dataset, explore_dataset, samples):
        monkeypatch.setattr(module, "load_dataset", lambda repo, split, num_proc=None: by_repo[(repo, split)])
    return fakes


@pytest.fixture
def hf_splits(hub):
    return hub["smart-turn"]
