"""Tests for ``pipeline.data``: deterministic path/label collection.

RED phase: ``pipeline.data`` is currently an empty stub, so this test fails
at runtime until Task 4 implements the module. Collection must still
succeed, hence the pipeline import is lazy (function-level).
"""


def test_collect_paths_deterministic(tmp_path):
    """collect_paths_labels must return the same (sorted) order for the same
    root, with labels aligned to the class subdirectories."""
    from pipeline.data import collect_paths_labels

    # Build a fake dataset root: root/{split}/{NORMAL,PNEUMONIA}/img_*.jpeg
    root = tmp_path / "dataset"
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            for i in range(3):
                d = root / split / cls
                d.mkdir(parents=True, exist_ok=True)
                (d / f"img_{i}.jpeg").write_bytes(b"\xff\xd8\xff\xe0")

    paths1, labels1 = collect_paths_labels(root, "train")
    paths2, labels2 = collect_paths_labels(root, "train")

    # Deterministic: two calls on the same root give identical results.
    assert paths1 == paths2
    assert labels1 == labels2

    # 3 NORMAL + 3 PNEUMONIA images in the train split.
    assert len(paths1) == 6
    assert len(labels1) == 6
    assert set(labels1) == {0, 1}

    # Sorted order -> deterministic regardless of filesystem enumeration.
    assert paths1 == sorted(paths1)