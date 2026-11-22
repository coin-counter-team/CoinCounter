"""CNN 비교용 동전 그림 만들기 (담당: 정준영).

동전별 정답이 없으므로, 정답이 곧 동전별 정답이 되는 사진(layout이 single·same)만 쓴다.
사진의 정답 종류가 하나뿐이고 검출된 동전 수가 정답 동전 수와 같을 때만 사용하고,
다르면 라벨을 믿을 수 없어 그 사진은 건너뛴다.

동전 하나를 반지름 기준으로 잘라 정사각형 그림(기본 64x64)으로 만든다.
반지름으로 맞추기 때문에 촬영 거리 차이는 사라지지만 크기 정보도 같이 사라진다.

같은 동전에 대해 규칙 기반 분류기(coin.classify) 예측도 같이 저장해,
CNN과 규칙 기반을 같은 동전에서 비교할 수 있게 한다.

실행 (저장소 루트에서):
    PYTHONPATH=src python eval/cnn_data.py \
        --labels ~/dev/coin_data/03_labels/coins.csv \
        --manifest ~/dev/coin_data/02_dataset/image_manifest.csv \
        --images data/raw --splits tune test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from make_preds import detect, index_images, read_image

from coin.classify import classify_all

ROOT = Path(__file__).resolve().parents[1]
VALUES = (10, 50, 100, 500)
LAYOUTS = {"single", "same"}


def load_labels(path: Path) -> dict[str, dict[int, int]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["파일"]: {v: int(r[f"{v}원"]) for v in VALUES} for r in csv.DictReader(f)}


def load_manifest(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["image"]: (r["split"], r["layout"]) for r in csv.DictReader(f)}


def single_value(counts: dict[int, int]) -> int | None:
    """정답이 한 종류뿐이면 그 금액, 아니면 None."""
    kinds = [v for v, n in counts.items() if n > 0]
    return kinds[0] if len(kinds) == 1 else None


def crop_coin(image: np.ndarray, x: int, y: int, r: int, size: int, margin: float) -> np.ndarray:
    """(x, y) 중심, 반지름 r의 동전을 r*margin 반변 정사각형으로 잘라 size로 줄인다."""
    half = max(int(round(r * margin)), 1)
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REPLICATE)
    patch = padded[y : y + 2 * half, x : x + 2 * half]  # 패딩만큼 밀렸으므로 좌표 그대로
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def build(labels, manifest, images, splits, size, margin):
    xs, ys, splits_out, names, idxs, rules = [], [], [], [], [], []
    for name, (split, layout) in sorted(manifest.items()):
        if split not in splits or layout not in LAYOUTS or name not in labels:
            continue
        counts = labels[name]
        value = single_value(counts)
        if value is None:
            print(f"건너뜀(종류가 여러 개): {name}", file=sys.stderr)
            continue
        path = images.get(name)
        if path is None:
            print(f"건너뜀(사진 없음): {name}", file=sys.stderr)
            continue
        image = read_image(path)
        circles = detect(image, "contour", "adaptive")
        if len(circles) != sum(counts.values()):
            print(
                f"건너뜀(검출 {len(circles)}개 != 정답 {sum(counts.values())}개): {name}",
                file=sys.stderr,
            )
            continue
        predicted = classify_all(image, circles)
        for i, ((x, y, r), pred) in enumerate(zip(circles, predicted, strict=True)):
            xs.append(crop_coin(image, x, y, r, size, margin))
            ys.append(value)
            splits_out.append(split)
            names.append(name)
            idxs.append(i)
            rules.append(-1 if pred is None else int(pred))
    return {
        "x": np.stack(xs) if xs else np.zeros((0, size, size, 3), np.uint8),
        "y": np.array(ys, dtype=np.int64),
        "split": np.array(splits_out),
        "image": np.array(names),
        "index": np.array(idxs, dtype=np.int64),
        "rule": np.array(rules, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN 비교용 동전 그림 만들기")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--splits", nargs="+", default=["tune", "test"])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.15)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/cnn_crops.npz")
    args = parser.parse_args()

    data = build(
        load_labels(args.labels),
        load_manifest(args.manifest),
        index_images(args.images),
        set(args.splits),
        args.size,
        args.margin,
    )
    if len(data["y"]) == 0:
        raise SystemExit("만들어진 동전 그림이 없습니다 (사진 위치와 split을 확인하세요)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    for split in sorted(set(data["split"])):
        mask = data["split"] == split
        per = {v: int((data["y"][mask] == v).sum()) for v in VALUES}
        print(f"[{split}] 동전 {int(mask.sum())}개 {per}")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
