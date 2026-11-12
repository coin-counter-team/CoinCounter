import csv
import importlib.util
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def load_eval_module(name: str):
    """eval/은 패키지가 아니므로 파일 경로로 직접 불러온다."""
    module_name = f"coin_eval_{name}"
    spec = importlib.util.spec_from_file_location(module_name, EVAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass가 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(module)
    return module


evaluate = load_eval_module("evaluate")
counts_eval = load_eval_module("evaluate_counts")
make_preds = load_eval_module("make_preds")


def write_csv(path: Path, header: list[str], rows: list[list], encoding: str = "utf-8") -> Path:
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


# ---------- evaluate.py: 동전 위치 정답 기반 평가 ----------

GT_HEADER = ["image", "split", "group", "condition", "x", "y", "r", "value"]
PRED_HEADER = ["image", "x", "y", "r", "value"]


def make_gt_files(tmp_path: Path, encoding: str = "utf-8"):
    labels = write_csv(
        tmp_path / "labels.csv",
        GT_HEADER,
        [
            ["a.jpg", "test", "g1", "base", 100, 100, 50, 100],
            ["a.jpg", "test", "g1", "base", 300, 100, 40, 10],
            ["b.jpg", "test", "g2", "base", 100, 100, 50, 500],
        ],
        encoding,
    )
    return evaluate.load_labels(labels)


def test_match_is_one_to_one():
    gts = [evaluate.Gt(100, 100, 50, 100)]
    preds = [evaluate.Pred(100, 100, 50, 100), evaluate.Pred(105, 100, 50, 100)]
    assert len(evaluate.match(gts, preds)) == 1


def test_match_respects_tolerance():
    gts = [evaluate.Gt(100, 100, 50, 100)]
    near = [evaluate.Pred(120, 100, 50, 100)]  # 중심 거리 20 <= 50 * 0.5
    far = [evaluate.Pred(140, 100, 50, 100)]  # 중심 거리 40 > 50 * 0.5
    assert len(evaluate.match(gts, near)) == 1
    assert evaluate.match(gts, far) == []


def test_predicted_total_skips_unclassified():
    preds = [evaluate.Pred(0, 0, 1, 100), evaluate.Pred(0, 0, 1, None)]
    assert evaluate.predicted_total(preds) == 100


def test_evaluate_perfect_predictions(tmp_path):
    meta, gt_coins = make_gt_files(tmp_path)
    preds = {
        "a.jpg": [evaluate.Pred(100, 100, 50, 100), evaluate.Pred(300, 100, 40, 10)],
        "b.jpg": [evaluate.Pred(100, 100, 50, 500)],
    }
    (row,) = evaluate.evaluate(meta, gt_coins, preds)
    assert row["detection_rate"] == "1.000"
    assert row["classification_acc"] == "1.000"
    assert row["total_match_rate"] == "1.000"
    assert row["pass"] == "True"


def test_evaluate_without_predictions_is_zero(tmp_path):
    meta, gt_coins = make_gt_files(tmp_path)
    (row,) = evaluate.evaluate(meta, gt_coins, {})
    assert row["detection_rate"] == "0.000"
    assert row["total_match_rate"] == "0.000"
    assert row["pass"] == "False"


def test_evaluate_wrong_class_and_extra_circle(tmp_path):
    meta, gt_coins = make_gt_files(tmp_path)
    preds = {
        "a.jpg": [evaluate.Pred(100, 100, 50, 100), evaluate.Pred(300, 100, 40, 10)],
        # 500원을 100원으로 오분류하고 엉뚱한 원도 하나 더 잡았다.
        "b.jpg": [evaluate.Pred(100, 100, 50, 100), evaluate.Pred(900, 900, 30, 50)],
    }
    (row,) = evaluate.evaluate(meta, gt_coins, preds)
    assert row["detection_rate"] == "1.000"
    assert row["classification_acc"] == "0.667"
    assert row["total_match_rate"] == "0.500"


def test_load_labels_and_preds_accept_bom(tmp_path):
    meta, gt_coins = make_gt_files(tmp_path, encoding="utf-8-sig")
    assert meta["a.jpg"] == ("test", "base")
    assert len(gt_coins["a.jpg"]) == 2

    preds_path = write_csv(
        tmp_path / "preds.csv", PRED_HEADER, [["a.jpg", 1, 2, 3, ""]], encoding="utf-8-sig"
    )
    assert evaluate.load_preds(preds_path)["a.jpg"][0].value is None


# ---------- evaluate_counts.py: 사진 단위 정답 기반 근사 평가 ----------

COUNT_HEADER = ["파일", "10원", "50원", "100원", "500원", "동전 수", "총액"]
MANIFEST_HEADER = ["image", "photographer", "layout", "split", "original_image"]


def make_count_files(tmp_path: Path, encoding: str = "utf-8-sig"):
    labels = write_csv(
        tmp_path / "coins.csv",
        COUNT_HEADER,
        [
            ["x_001.jpg", 1, 0, 1, 0, 2, 110],
            ["x_002.jpg", 0, 0, 0, 2, 2, 1000],
            ["x_003.jpg", 0, 1, 0, 0, 1, 50],
        ],
        encoding,
    )
    manifest = write_csv(
        tmp_path / "manifest.csv",
        MANIFEST_HEADER,
        [
            ["x_001.jpg", "a", "mixed", "tune", "o1.jpg"],
            ["x_002.jpg", "a", "same", "tune", "o2.jpg"],
            ["x_003.jpg", "b", "single", "test", "o3.jpg"],
        ],
        encoding,
    )
    return counts_eval.load_counts(labels), counts_eval.load_manifest(manifest)


PERFECT_PREDS = {
    "x_001.jpg": [10, 100],
    "x_002.jpg": [500, 500],
    "x_003.jpg": [50],
}


def by_key(rows):
    return {(r["split"], r["layout"]): r for r in rows}


def test_counts_load_handles_bom(tmp_path):
    counts, meta = make_count_files(tmp_path)
    assert counts["x_001.jpg"] == {10: 1, 50: 0, 100: 1, 500: 0}
    assert meta["x_003.jpg"] == ("test", "single")


def test_counts_perfect_predictions(tmp_path):
    counts, meta = make_count_files(tmp_path)
    rows = by_key(counts_eval.evaluate(counts, meta, PERFECT_PREDS))
    for row in rows.values():
        assert row["detect_approx"] == "1.000"
        assert row["classify_approx"] == "1.000"
        assert row["total_match_rate"] == "1.000"
        assert row["count_exact_rate"] == "1.000"
    assert rows[("tune", "all")]["images"] == 2


def test_counts_without_predictions_is_zero(tmp_path):
    counts, meta = make_count_files(tmp_path)
    rows = counts_eval.evaluate(counts, meta, {})
    assert all(r["detect_approx"] == "0.000" for r in rows)
    assert all(r["total_match_rate"] == "0.000" for r in rows)


def test_counts_wrong_class_breaks_total_but_not_detection(tmp_path):
    counts, meta = make_count_files(tmp_path)
    preds = dict(PERFECT_PREDS, **{"x_002.jpg": [500, 100]})  # 500원 하나를 100원으로
    row = by_key(counts_eval.evaluate(counts, meta, preds))[("tune", "same")]
    assert row["detect_approx"] == "1.000"
    assert row["classify_approx"] == "0.500"
    assert row["total_match_rate"] == "0.000"
    assert row["count_exact_rate"] == "0.000"


def test_counts_unclassified_is_excluded_from_total(tmp_path):
    counts, meta = make_count_files(tmp_path)
    preds = dict(PERFECT_PREDS, **{"x_001.jpg": [10, None]})
    row = by_key(counts_eval.evaluate(counts, meta, preds))[("tune", "mixed")]
    assert row["detect_approx"] == "1.000"  # 동전 수는 맞게 찾았다
    assert row["total_match_rate"] == "0.000"  # 100원이 빠져 총액이 다르다


def test_counts_splits_filter(tmp_path):
    counts, meta = make_count_files(tmp_path)
    rows = counts_eval.evaluate(counts, meta, PERFECT_PREDS, {"test"})
    assert {r["split"] for r in rows} == {"test"}


def test_counts_extra_circles_do_not_raise_detection_above_one(tmp_path):
    counts, meta = make_count_files(tmp_path)
    preds = dict(PERFECT_PREDS, **{"x_003.jpg": [50, 50, 50]})
    row = by_key(counts_eval.evaluate(counts, meta, preds))[("test", "single")]
    assert row["detect_approx"] == "1.000"
    assert row["total_match_rate"] == "0.000"


def test_counts_write_summary(tmp_path):
    counts, meta = make_count_files(tmp_path)
    rows = counts_eval.evaluate(counts, meta, PERFECT_PREDS)
    out = tmp_path / "out" / "summary.csv"
    counts_eval.write_summary(rows, out)
    with out.open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)


# ---------- make_preds.py ----------


def test_target_images_reads_bom_manifest_and_filters_split(tmp_path):
    manifest = write_csv(
        tmp_path / "manifest.csv",
        MANIFEST_HEADER,
        [
            ["a.jpg", "p", "single", "tune", "o1.jpg"],
            ["b.jpg", "p", "single", "test", "o2.jpg"],
            ["a.jpg", "p", "single", "tune", "o1.jpg"],  # 같은 이미지가 다시 나와도 한 번만
        ],
        encoding="utf-8-sig",
    )
    assert make_preds.target_images(manifest, {"tune"}) == ["a.jpg"]
    assert make_preds.target_images(manifest, {"tune", "test"}) == ["a.jpg", "b.jpg"]


def test_index_images_finds_nested_files(tmp_path):
    nested = tmp_path / "tune" / "single"
    nested.mkdir(parents=True)
    (nested / "x_001.jpg").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore")
    assert set(make_preds.index_images(tmp_path)) == {"x_001.jpg"}


def test_predict_without_classifier_leaves_value_empty(monkeypatch):
    monkeypatch.setattr(make_preds, "detect", lambda *_: [(10, 20, 30), (40, 50, 60)])
    result = make_preds.predict(None, "contour", "adaptive", None)
    assert result == [(10, 20, 30, None), (40, 50, 60, None)]


def test_predict_rejects_mismatched_classifier_output(monkeypatch):
    monkeypatch.setattr(make_preds, "detect", lambda *_: [(10, 20, 30), (40, 50, 60)])
    with pytest.raises(ValueError):
        make_preds.predict(None, "contour", "adaptive", lambda image, circles: [100])
