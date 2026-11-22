import importlib.util
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def load_cnn_data():
    """cnn_data.py가 같은 폴더의 make_preds를 불러오므로 eval/을 경로에 넣고 읽는다."""
    sys.path.insert(0, str(EVAL_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "coin_eval_cnn_data", EVAL_DIR / "cnn_data.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["coin_eval_cnn_data"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EVAL_DIR))
    return module


cnn_data = load_cnn_data()

ZERO = {10: 0, 50: 0, 100: 0, 500: 0}


def test_single_value_returns_only_kind():
    assert cnn_data.single_value({**ZERO, 100: 5}) == 100


def test_single_value_none_when_mixed_or_empty():
    assert cnn_data.single_value({**ZERO, 10: 1, 50: 2}) is None
    assert cnn_data.single_value(ZERO) is None


def test_crop_coin_size_and_center():
    image = np.zeros((200, 200, 3), np.uint8)
    image[90:110, 90:110] = 255  # 중심 (100, 100)의 흰 점
    patch = cnn_data.crop_coin(image, 100, 100, 40, size=32, margin=1.0)
    assert patch.shape == (32, 32, 3)
    assert patch[16, 16].min() == 255  # 가운데가 동전 중심
    assert patch[0, 0].max() == 0


def test_crop_coin_near_edge_keeps_size():
    image = np.full((100, 100, 3), 120, np.uint8)
    patch = cnn_data.crop_coin(image, 5, 5, 30, size=64, margin=1.15)
    assert patch.shape == (64, 64, 3)
    assert (patch == 120).all()  # 가장자리 밖은 테두리 색으로 채운다


def test_load_labels_handles_bom(tmp_path):
    path = tmp_path / "coins.csv"
    path.write_text("파일,10원,50원,100원,500원\na.jpg,0,0,3,0\n", encoding="utf-8-sig")
    assert cnn_data.load_labels(path) == {"a.jpg": {10: 0, 50: 0, 100: 3, 500: 0}}


def run_build(monkeypatch, labels, manifest, circles, tmp_path, splits=("tune",)):
    image = np.full((300, 300, 3), 100, np.uint8)
    monkeypatch.setattr(cnn_data, "read_image", lambda path: image)
    monkeypatch.setattr(cnn_data, "detect", lambda img, detector, pre: circles)
    monkeypatch.setattr(cnn_data, "classify_all", lambda img, cs: [100] * len(cs))
    images = {name: tmp_path / name for name in labels}
    return cnn_data.build(labels, manifest, images, set(splits), size=16, margin=1.15)


def test_build_labels_every_coin_with_photo_value(monkeypatch, tmp_path):
    labels = {"same_1.jpg": {**ZERO, 50: 2}}
    manifest = {"same_1.jpg": ("tune", "same")}
    circles = [(60, 60, 20), (200, 200, 20)]
    data = run_build(monkeypatch, labels, manifest, circles, tmp_path)
    assert data["x"].shape == (2, 16, 16, 3)
    assert list(data["y"]) == [50, 50]
    assert list(data["rule"]) == [100, 100]
    assert list(data["split"]) == ["tune", "tune"]


def test_build_skips_mixed_wrong_count_and_other_splits(monkeypatch, tmp_path):
    labels = {
        "mixed.jpg": {**ZERO, 10: 1, 100: 1},
        "wrong.jpg": {**ZERO, 500: 3},
        "test_same.jpg": {**ZERO, 10: 1},
    }
    manifest = {
        "mixed.jpg": ("tune", "mixed_large"),
        "wrong.jpg": ("tune", "same"),
        "test_same.jpg": ("test", "same"),
    }
    data = run_build(monkeypatch, labels, manifest, [(60, 60, 20)], tmp_path)
    assert len(data["y"]) == 0  # 섞인 사진, 개수 불일치, 다른 split 모두 제외


def test_build_marks_classification_failure(monkeypatch, tmp_path):
    labels = {"single_1.jpg": {**ZERO, 500: 1}}
    manifest = {"single_1.jpg": ("tune", "single")}
    monkeypatch.setattr(cnn_data, "read_image", lambda path: np.zeros((200, 200, 3), np.uint8))
    monkeypatch.setattr(cnn_data, "detect", lambda img, detector, pre: [(100, 100, 30)])
    monkeypatch.setattr(cnn_data, "classify_all", lambda img, cs: [None])
    data = cnn_data.build(
        labels, manifest, {"single_1.jpg": tmp_path / "x"}, {"tune"}, size=16, margin=1.15
    )
    assert list(data["rule"]) == [-1]
