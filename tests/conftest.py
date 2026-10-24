import pytest

from turn_detection.data import dataset
from turn_detection.data.constants import DATASETS

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
    by_repo = {DATASETS[name]["splits"][split]: ds for name, splits in fakes.items() for split, ds in splits.items()}
    monkeypatch.setattr(dataset, "load_dataset", lambda repo, split="train", num_proc=None: by_repo[(repo, split)])
    return fakes


@pytest.fixture
def hf_splits(hub):
    return hub["smart-turn"]


@pytest.fixture
def ttd_splits(hub):
    return hub["ttd"]


@pytest.fixture
def artifacts(tmp_path, hf_splits, monkeypatch):
    """scripts/split.py outputs for the fake smart-turn splits: artifacts/manifests/."""
    import split

    root = tmp_path / "artifacts"
    monkeypatch.setattr("sys.argv", ["split.py", "--out", str(root / "manifests")])
    split.main()
    return root
