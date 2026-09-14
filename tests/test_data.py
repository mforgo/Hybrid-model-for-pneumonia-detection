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


def test_extract_patient_group_id_namespaced():
    """Group IDs must combine patient id AND etiology: person100_bacteria is
    a different patient from person100_virus (Kermany id reuse)."""
    from pipeline.data import extract_patient_group_id

    from pathlib import Path

    # Canonical Kermany filename -> namespaced group id.
    assert (
        extract_patient_group_id("person369_bacteria_1680.jpeg")
        == "person369_bacteria"
    )
    assert (
        extract_patient_group_id("person369_virus_1680.jpeg") == "person369_virus"
    )

    # Same numeric id, different etiology -> DIFFERENT groups (id reuse fix).
    assert extract_patient_group_id("person100_bacteria_1.jpeg") != (
        extract_patient_group_id("person100_virus_2.jpeg")
    )

    # Same numeric id + same etiology -> same group regardless of suffix.
    assert extract_patient_group_id("person100_bacteria_1.jpeg") == (
        extract_patient_group_id("person100_bacteria_2.jpeg")
    )

    # Path objects and deep directory nesting work too.
    assert (
        extract_patient_group_id(Path("train/PNEUMONIA/person7_bacteria_3.jpeg"))
        == "person7_bacteria"
    )

    # Files that do not match the Kermany pattern become singleton groups.
    assert extract_patient_group_id("NORMAL2-IM-1427-0001.jpeg") == (
        "NORMAL2-IM-1427-0001"
    )


def test_group_split_data_patient_disjoint():
    """group_split_data must never place the same patient group in both
    train and validation."""
    from pipeline.data import extract_patient_group_id, group_split_data

    from pathlib import Path

    # 20 patients x 4 images each, alternating pneumonia/normal at folder
    # level (PNEUMONIA/NORMAL), mimicking Kermany naming.
    paths: list[Path] = []
    labels: list[int] = []
    for patient in range(1, 21):
        for img in range(4):
            cls = "NORMAL" if patient % 2 == 0 else "PNEUMONIA"
            label = 1 if cls == "PNEUMONIA" else 0
            etiology = "bacteria" if cls == "PNEUMONIA" else "virus"
            name = f"person{patient}_{etiology}_{img}.jpeg"
            paths.append(Path(f"dataset/train/{cls}/{name}"))
            labels.append(label)

    groups = [extract_patient_group_id(p) for p in paths]
    train_paths, val_paths, train_labels, val_labels = group_split_data(
        paths, labels, groups, val_split=0.20, seed=6
    )

    # All images accounted for, labels aligned.
    assert len(train_paths) + len(val_paths) == len(paths)
    assert len(train_paths) == len(train_labels)
    assert len(val_paths) == len(val_labels)

    # Patient-disjointness: no group id appears in both splits.
    train_groups = {extract_patient_group_id(p) for p in train_paths}
    val_groups = {extract_patient_group_id(p) for p in val_paths}
    assert train_groups.isdisjoint(val_groups)

    # Both classes survive the split.
    assert 0 in train_labels and 1 in train_labels
    assert 0 in val_labels and 1 in val_labels


def test_group_split_data_deterministic():
    """The same seed must reproduce the identical split."""
    from pipeline.data import extract_patient_group_id, group_split_data

    from pathlib import Path

    paths = [Path(f"p/n/person{i}_bacteria_{j}.jpeg") for i in range(1, 11) for j in range(4)]
    labels = [1 if i % 2 else 0 for i in range(1, 11) for _ in range(4)]
    groups = [extract_patient_group_id(p) for p in paths]

    t1, v1, _, _ = group_split_data(paths, labels, groups, val_split=0.20, seed=6)
    t2, v2, _, _ = group_split_data(paths, labels, groups, val_split=0.20, seed=6)

    assert t1 == t2
    assert v1 == v2


def test_split_data_backward_compatible():
    """split_data (per-image stratified) still works for split_strategy=random."""
    from pipeline.data import split_data

    from pathlib import Path

    paths = [Path(f"p/n/img_{i}.jpeg") for i in range(100)]
    labels = [1 if i % 3 else 0 for i in range(100)]  # ~2:1 pneumonia:normal

    train_paths, val_paths, train_labels, val_labels = split_data(
        paths, labels, val_split=0.20, seed=6
    )

    assert len(train_paths) + len(val_paths) == 100
    assert set(train_labels) == {0, 1}
    assert set(val_labels) == {0, 1}
    # Stratified: both classes keep their ratio in each split.
    def ratio(ls):
        return sum(1 for x in ls if x == 1) / len(ls)

    assert abs(ratio(train_labels) - ratio(labels)) < 0.05
    assert abs(ratio(val_labels) - ratio(labels)) < 0.05