import pandas as pd

from split import build_splits

from .helpers import make_meta


def test_train_val_is_a_cut_by_id():
    meta = make_meta()
    splits = build_splits(meta, make_meta([("tts_a", True, 20)], seed=9), val_frac=0.1)
    train, val = splits["train"], splits["val"]
    assert len(train) + len(val) == len(meta) and len(val) == round(len(meta) * 0.1)
    assert train["id"].max() < val["id"].min()  # val = the highest ids


def test_split_does_not_depend_on_row_order():
    meta, test = make_meta(), make_meta([("tts_a", True, 20)], seed=9)
    a = build_splits(meta, test)["val"]
    b = build_splits(meta.sample(frac=1, random_state=0), test)["val"]
    assert a["id"].tolist() == b["id"].tolist()


def test_no_id_overlap_between_train_val_test():
    full_train = make_meta(seed=0)
    test = pd.concat([make_meta([("tts_a", True, 50)], seed=1), full_train.iloc[:3]], ignore_index=True)
    splits = build_splits(full_train, test)  # 3 ids also in the test split: dropped from train/val
    ids = {name: set(df["id"]) for name, df in splits.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert len(ids["train"]) + len(ids["val"]) == len(full_train) - 3
