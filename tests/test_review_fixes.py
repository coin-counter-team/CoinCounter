import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from coin.detection_metrics import detection_counts
from coin.image_io import read_image

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).parents[1] / "tools/compare_detection.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)
    return str(path)


def test_unicode_image_and_read_failures(tmp_path):
    path = tmp_path / "한글 사진.png"
    expected = np.full((20, 30, 3), 123, dtype=np.uint8)
    cv2.imencode(".png", expected)[1].tofile(path)
    assert np.array_equal(read_image(path), expected)
    for name, data in [("empty.png", b""), ("broken.png", b"not an image")]:
        bad = tmp_path / name
        bad.write_bytes(data)
        with pytest.raises(ValueError, match=name):
            read_image(bad)
    with pytest.raises(ValueError):
        read_image(tmp_path / "missing.png")


def test_explicit_empty_image_is_evaluated(tmp_path):
    path = write_csv(
        tmp_path / "labels.csv",
        [
            ["image", "x", "y", "r", "split", "group"],
            ["empty.jpg", "", "", "", "test", "g1"],
            ["coin.jpg", 50, 50, 20, "test", "g2"],
        ],
    )
    truth = comparison.load_truth(path)
    assert comparison.load_counts(path) == {"empty.jpg": 0, "coin.jpg": 1}
    assert detection_counts([(10, 10, 5)], truth["empty.jpg"]) == (0, 1, 0)


def test_count_only_zero_is_truth_but_positive_count_is_not(tmp_path):
    path = write_csv(
        tmp_path / "labels.csv",
        [
            ["파일", "동전 수"],
            ["empty.jpg", 0],
            ["coin.jpg", 1],
        ],
    )
    assert comparison.load_truth(path) == {"empty.jpg": []}


def test_split_selection_and_group_leakage(tmp_path):
    rows = [["image", "split", "group"], ["a.jpg", "tune", "g1"], ["b.jpg", "test", "g2"]]
    path = write_csv(tmp_path / "labels.csv", rows)
    paths = [Path("a.jpg"), Path("b.jpg")]
    with pytest.raises(ValueError, match="split"):
        comparison.select_images(paths, path, None)
    assert comparison.select_images(paths, path, "test") == [Path("b.jpg")]
    with pytest.raises(ValueError):
        comparison.select_images(paths[:1], path, "test")
    rows[2][2] = "g1"
    write_csv(Path(path), rows)
    with pytest.raises(ValueError, match="group"):
        comparison.select_images(paths, path, "test")


def test_contour_preprocessing_preserves_detection():
    image = np.full((200, 200, 3), 240, np.uint8)
    cv2.circle(image, (100, 100), 30, (20, 20, 20), -1)
    small, gray, scales = comparison.prepare(image, 200, "light")
    mask = comparison.contour_mask(gray, "light")
    assert scales == (1, 1)
    assert comparison.detect_contour(mask) == [(100, 100, 30)]
