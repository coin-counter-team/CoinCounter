"""예측 CSV 생성 (담당: 정준영).

정답 CSV에 있는 이미지를 전처리 → 검출 → (분류) 순서로 처리해
evaluate.py가 읽는 예측 CSV(image,x,y,r,value)를 만든다.

--labels에는 image, split 열이 있는 CSV면 무엇이든 넘길 수 있다
(예: 사진 단위 정답만 있을 때는 image_manifest.csv).

분류가 아직 구현되지 않았다면 --classifier none 으로 검출률만 먼저 평가할 수 있다.
이때 value 열은 비워 두며, 분류 정확도와 총금액 일치율은 의미가 없다.

실행 (저장소 루트에서):
    PYTHONPATH=src python3 eval/make_preds.py --splits test \
        --detector contour --classifier none --out eval/results/preds.csv
    python3 eval/evaluate.py --preds eval/results/preds.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from coin.detect import Circle, detect_contour, detect_hough
from coin.preprocess import preprocess_image, to_gray

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

Classifier = Callable[[np.ndarray, list[Circle]], list]


def read_image(path: Path) -> np.ndarray:
    """한글 경로에서도 읽히도록 바이트로 읽어 디코딩한다."""
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if image is None:
        raise OSError(f"사진을 읽을 수 없습니다: {path}")
    return image


def index_images(root: Path) -> dict[str, Path]:
    """root 아래 모든 사진을 파일 이름 → 경로로 모은다 (하위 폴더 포함)."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in found:
                print(f"경고: 같은 이름의 사진이 둘 이상입니다: {path.name}", file=sys.stderr)
                continue
            found[path.name] = path
    return found


def target_images(labels: Path, splits: set[str]) -> list[str]:
    """정답 CSV에서 대상 split의 이미지 이름을 순서대로 중복 없이 뽑는다."""
    names: list[str] = []
    seen: set[str] = set()
    with labels.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] in splits and row["image"] not in seen:
                seen.add(row["image"])
                names.append(row["image"])
    return names


def detect(image: np.ndarray, detector: str, preprocess: str) -> list[Circle]:
    if detector == "contour":
        return detect_contour(preprocess_image(image, preprocess))
    if detector == "hough":
        return detect_hough(to_gray(image))
    raise ValueError(f"지원하지 않는 검출 방식: {detector}")


def load_classifier(name: str) -> Classifier | None:
    if name == "none":
        return None
    if name == "rule":
        from coin.classify import classify_all

        return classify_all
    raise ValueError(f"지원하지 않는 분류 방식: {name}")


def predict(
    image: np.ndarray,
    detector: str,
    preprocess: str,
    classifier: Classifier | None,
) -> list[tuple[int, int, int, int | None]]:
    circles = detect(image, detector, preprocess)
    if classifier is None:
        values: list = [None] * len(circles)
    else:
        values = classifier(image, circles)
        if len(values) != len(circles):
            raise ValueError("분류 결과 개수가 검출된 동전 개수와 다릅니다")
    return [(x, y, r, v) for (x, y, r), v in zip(circles, values, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="평가용 예측 CSV 생성")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/labels/coins.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["tune", "test", "ext"])
    parser.add_argument("--detector", choices=["contour", "hough"], default="contour")
    parser.add_argument("--preprocess", choices=["adaptive", "otsu"], default="adaptive")
    parser.add_argument("--classifier", choices=["none", "rule"], default="none")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/preds.csv")
    args = parser.parse_args()

    names = target_images(args.labels, set(args.splits))
    if not names:
        raise SystemExit(f"대상 이미지가 없습니다: splits={args.splits}")
    paths = index_images(args.images)
    classifier = load_classifier(args.classifier)

    rows = []
    missing = 0
    for name in names:
        path = paths.get(name)
        if path is None:
            print(f"경고: 사진을 찾을 수 없어 건너뜁니다: {name}", file=sys.stderr)
            missing += 1
            continue
        try:
            preds = predict(read_image(path), args.detector, args.preprocess, classifier)
        except NotImplementedError:
            raise SystemExit(
                "분류가 아직 구현되지 않았습니다. --classifier none 으로 실행하세요."
            ) from None
        rows.extend((name, x, y, r, "" if v is None else v) for x, y, r, v in preds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x", "y", "r", "value"])
        writer.writerows(rows)
    print(f"이미지 {len(names) - missing}/{len(names)}장, 동전 {len(rows)}개 저장: {args.out}")


if __name__ == "__main__":
    main()
