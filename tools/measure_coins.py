"""정답 라벨(data/labels/coins.csv)로 동전별 반지름·HSV를 출력해 임계값을 정한다.

사용법 (저장소 루트에서):
    PYTHONPATH=src python tools/measure_coins.py data/labels/coins.csv data/raw --split tune

출력: 동전마다 지름 px, px_per_mm, 평균 H·S·V, 구리 비율. 마지막에 금액별 요약.
여기 나온 값을 classify.py의 PX_PER_MM, COPPER_* 상수에 직접 옮겨 적는다.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from coin.classify import COIN_DIAMETER_MM, copper_ratio
from coin.features import crop_coin, hsv_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", help="coins.csv 경로")
    parser.add_argument("image_dir", help="사진 폴더 (예: data/raw)")
    parser.add_argument("--split", default="tune", help="사용할 split (기본 tune)")
    args = parser.parse_args()

    by_value: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    print("image,value,diam_px,px_per_mm,H,S,V,copper_ratio")
    with open(args.labels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != args.split:
                continue
            image = cv2.imread(str(Path(args.image_dir) / row["image"]))
            if image is None:
                print(f"# 읽기 실패: {row['image']}")
                continue
            value = int(row["value"])
            circle = (float(row["x"]), float(row["y"]), float(row["r"]))
            crop = crop_coin(image, circle)
            h, s, v = hsv_features(crop)
            ppm = 2 * circle[2] / COIN_DIAMETER_MM[value]
            ratio = copper_ratio(crop)
            by_value[value].append((ppm, h, s, ratio))
            d = 2 * circle[2]
            print(f"{row['image']},{value},{d:.1f},{ppm:.3f},{h:.1f},{s:.1f},{v:.1f},{ratio:.2f}")

    print("\n# 금액별 요약: 개수 / px_per_mm 중앙값 / H·S·구리비율 최소~최대")
    all_ppm = []
    for value in sorted(by_value):
        rows = by_value[value]
        ppm, hs, ss, rs = (list(col) for col in zip(*rows, strict=True))
        all_ppm += ppm
        print(
            f"{value}원: {len(rows)}개 / {statistics.median(ppm):.3f} / "
            f"H {min(hs):.1f}~{max(hs):.1f} / S {min(ss):.1f}~{max(ss):.1f} / "
            f"구리 {min(rs):.2f}~{max(rs):.2f}"
        )
    if all_ppm:
        print(f"\n# 전체 px_per_mm 중앙값 → PX_PER_MM = {statistics.median(all_ppm):.3f}")


if __name__ == "__main__":
    main()
