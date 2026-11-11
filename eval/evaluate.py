"""평가 모듈 (담당: 정준영).

정답 CSV(data/labels/coins.csv)와 예측 CSV를 비교해
검출률·분류 정확도·총금액 일치율을 (split, condition)별로 계산한다.

예측 CSV 형식 (동전 한 개당 한 줄, 정답 CSV와 같은 열 이름 사용):
    image,x,y,r,value
    base_g03_012.jpg,410,283,56,100
value가 비어 있으면 분류 실패(None)로 보고 총금액에서 제외한다.

실행:
    python eval/evaluate.py --labels data/labels/coins.csv \
        --preds eval/results/preds.csv --out eval/results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# (split, condition)별 목표 지표. 제안서 기준: 기본 85%, 확장 80%.
TARGETS = {"test": 0.85, "ext": 0.80}

# 중심 거리가 정답 반지름의 이 비율 이내면 같은 동전으로 본다 (팀 합의 필요).
MATCH_TOLERANCE = 0.5


@dataclass(frozen=True)
class Gt:
    x: float
    y: float
    r: float
    value: int


@dataclass(frozen=True)
class Pred:
    x: float
    y: float
    r: float
    value: int | None


def load_labels(path: Path):
    """정답 CSV를 읽어 이미지별 (메타 정보, 동전 목록)으로 반환한다."""
    meta: dict[str, tuple[str, str]] = {}
    coins: dict[str, list[Gt]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image = row["image"]
            meta[image] = (row["split"], row["condition"])
            coins[image].append(
                Gt(float(row["x"]), float(row["y"]), float(row["r"]), int(row["value"]))
            )
    return meta, coins


def load_preds(path: Path) -> dict[str, list[Pred]]:
    """예측 CSV를 읽어 이미지별 동전 목록으로 반환한다."""
    preds: dict[str, list[Pred]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("value") or "").strip()
            preds[row["image"]].append(
                Pred(
                    float(row["x"]),
                    float(row["y"]),
                    float(row["r"]),
                    int(raw) if raw else None,
                )
            )
    return preds


def match(gts: list[Gt], preds: list[Pred], tol: float = MATCH_TOLERANCE) -> list[tuple[Gt, Pred]]:
    """중심 거리가 가까운 순서로 정답-예측을 일대일 매칭한다 (그리디)."""
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            dist = math.hypot(g.x - p.x, g.y - p.y)
            if dist <= tol * g.r:
                pairs.append((dist, gi, pi))
    pairs.sort()

    used_g: set[int] = set()
    used_p: set[int] = set()
    matched = []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gts[gi], preds[pi]))
    return matched


def predicted_total(preds: list[Pred]) -> int:
    """예측된 총금액. 분류 실패(None)는 제외한다 (count.py 규칙과 동일)."""
    return sum(p.value for p in preds if p.value is not None)


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def evaluate(meta, gt_coins, preds):
    """(split, condition)별 지표를 계산해 행 목록으로 반환한다."""
    acc = defaultdict(lambda: defaultdict(int))
    for image, gts in gt_coins.items():
        key = meta[image]
        image_preds = preds.get(image, [])
        matched = match(gts, image_preds)

        acc[key]["images"] += 1
        acc[key]["gt_coins"] += len(gts)
        acc[key]["detected"] += len(matched)
        acc[key]["correct"] += sum(1 for g, p in matched if g.value == p.value)
        gt_total = sum(g.value for g in gts)
        acc[key]["total_ok"] += int(predicted_total(image_preds) == gt_total)

    rows = []
    for (split, condition), a in sorted(acc.items()):
        det = _rate(a["detected"], a["gt_coins"])
        cls = _rate(a["correct"], a["detected"])
        tot = _rate(a["total_ok"], a["images"])
        target = TARGETS.get(split)
        passed = "" if target is None else str(min(det, cls, tot) >= target)
        rows.append(
            {
                "split": split,
                "condition": condition,
                "images": a["images"],
                "gt_coins": a["gt_coins"],
                "detection_rate": f"{det:.3f}",
                "classification_acc": f"{cls:.3f}",
                "total_match_rate": f"{tot:.3f}",
                "target": "" if target is None else f"{target:.2f}",
                "pass": passed,
            }
        )
    return rows


def write_summary(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="동전 검출·분류 평가")
    parser.add_argument("--labels", type=Path, default=Path("data/labels/coins.csv"))
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=["tune", "test", "ext"], help="평가할 split (기본: 전부)"
    )
    parser.add_argument("--out", type=Path, default=Path("eval/results/summary.csv"))
    args = parser.parse_args()

    meta, gt_coins = load_labels(args.labels)
    if args.splits:
        gt_coins = {i: c for i, c in gt_coins.items() if meta[i][0] in args.splits}
    rows = evaluate(meta, gt_coins, load_preds(args.preds))
    if not rows:
        raise SystemExit("평가할 정답 데이터가 없습니다")

    for row in rows:
        print(
            f"[{row['split']}/{row['condition']}] 이미지 {row['images']}장 | "
            f"검출률 {row['detection_rate']} | 분류 {row['classification_acc']} | "
            f"총금액 {row['total_match_rate']} | 통과 {row['pass'] or '-'}"
        )
    write_summary(rows, args.out)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
