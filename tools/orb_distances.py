"""정답 라벨로 '같은 금액끼리' vs '다른 금액끼리' ORB 해밍 거리 분포를 출력한다.

사용법 (저장소 루트에서):
    PYTHONPATH=src python tools/orb_distances.py data/labels/coins.csv data/raw --split tune

같은 금액 쌍의 거리는 작고, 다른 금액 쌍은 커야 한다. 두 분포가 갈리는 지점을
features.py의 ORB_MAX_DISTANCE로 정한다.
"""

import argparse
import csv
import itertools
from pathlib import Path

import cv2
import numpy as np

from coin.features import crop_coin, orb_descriptors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("image_dir")
    parser.add_argument("--split", default="tune")
    parser.add_argument("--max-coins", type=int, default=40, help="쌍 개수 제한용")
    args = parser.parse_args()

    coins: list[tuple[int, np.ndarray]] = []
    with open(args.labels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != args.split or len(coins) >= args.max_coins:
                continue
            image = cv2.imread(str(Path(args.image_dir) / row["image"]))
            if image is None:
                continue
            circle = (float(row["x"]), float(row["y"]), float(row["r"]))
            des = orb_descriptors(crop_coin(image, circle))
            if des is not None:
                coins.append((int(row["value"]), des))

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    same: list[float] = []
    diff: list[float] = []
    for (v1, d1), (v2, d2) in itertools.combinations(coins, 2):
        dists = [m.distance for m in matcher.match(d1, d2)]
        (same if v1 == v2 else diff).extend(dists)

    print(f"동전 {len(coins)}개")
    for name, data in (("같은 금액", same), ("다른 금액", diff)):
        if data:
            p = np.percentile(data, [10, 25, 50, 75, 90])
            print(f"{name}: 매칭 {len(data)}개, 10/25/50/75/90% = {np.round(p, 1).tolist()}")
    print("거리 기준 후보별 '좋은 매칭' 비율 (같은 금액이 높고 다른 금액이 낮을수록 좋음)")
    for t in (30, 40, 50, 64, 80):
        s = np.mean(np.array(same) <= t) if same else 0.0
        d = np.mean(np.array(diff) <= t) if diff else 0.0
        print(f"  ≤{t}: 같은 금액 {s:.2f} / 다른 금액 {d:.2f}")


if __name__ == "__main__":
    main()
