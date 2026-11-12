"""이미지 단위 평가 (담당: 정준영).

정답이 '사진 한 장당 한 줄'(종류별 개수·총액)인 경우를 위한 평가 스크립트다.
동전의 위치(x, y, r) 정답이 없으므로 evaluate.py의 위치 기반 검출률 대신
개수 비교로 얻은 근사치를 계산한다. 위치 정답이 생기면 evaluate.py를 쓴다.

입력
  --labels    정답 CSV. 열: 파일, 10원, 50원, 100원, 500원, 동전 수, 총액
  --manifest  image_manifest.csv. 열: image, photographer, layout, split, original_image
  --preds     예측 CSV (make_preds.py 결과). 열: image, x, y, r, value
              value가 비어 있으면 분류 실패로 보고 총금액에서 제외한다.

지표 (모두 근사치)
  total_match_rate    예측 총액이 정답 총액과 같은 이미지의 비율
  count_exact_rate    종류별 개수가 모두 정답과 같은 이미지의 비율
  detect_approx       Σmin(예측 동전 수, 정답 동전 수) / Σ정답 동전 수
  classify_approx     Σ종류별 min(예측, 정답) / Σmin(예측 동전 수, 정답 동전 수)

실행:
    python eval/evaluate_counts.py --labels data/labels/coins.csv \
        --manifest data/image_manifest.csv --preds eval/results/preds.csv \
        --splits tune --out eval/results/summary_counts.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

VALUES = (10, 50, 100, 500)


def load_counts(path: Path) -> dict[str, dict[int, int]]:
    """정답 CSV를 이미지 이름 → {금액: 개수}로 읽는다."""
    counts: dict[str, dict[int, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["파일"]] = {v: int(row[f"{v}원"]) for v in VALUES}
    return counts


def load_manifest(path: Path) -> dict[str, tuple[str, str]]:
    """매니페스트를 이미지 이름 → (split, layout)으로 읽는다."""
    meta: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            meta[row["image"]] = (row["split"], row["layout"])
    return meta


def load_pred_values(path: Path) -> dict[str, list[int | None]]:
    """예측 CSV를 이미지 이름 → 동전별 예측 금액 목록(None은 분류 실패)으로 읽는다."""
    preds: dict[str, list[int | None]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("value") or "").strip()
            preds[row["image"]].append(int(raw) if raw else None)
    return preds


def _rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate(counts, meta, preds, splits=None):
    """(split, layout)별 근사 지표를 계산한다. layout 'all'은 split 전체 합계다."""
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for image, gt in counts.items():
        if image not in meta:
            continue
        split, layout = meta[image]
        if splits and split not in splits:
            continue

        values = preds.get(image, [])
        known = [v for v in values if v is not None]
        pred = {v: known.count(v) for v in VALUES}
        n_gt = sum(gt.values())
        n_pred = len(values)
        overlap = sum(min(pred[v], gt[v]) for v in VALUES)

        for key in ((split, layout), (split, "all")):
            a = acc[key]
            a["images"] += 1
            a["gt_coins"] += n_gt
            a["found"] += min(n_pred, n_gt)
            a["typed"] += overlap
            a["total_ok"] += int(sum(known) == sum(v * n for v, n in gt.items()))
            a["count_ok"] += int(all(pred[v] == gt[v] for v in VALUES))

    def order(item):
        (split, layout), _ = item
        return split, layout != "all", layout  # split별로 'all'을 맨 위에

    rows = []
    for (split, layout), a in sorted(acc.items(), key=order):
        rows.append(
            {
                "split": split,
                "layout": layout,
                "images": int(a["images"]),
                "gt_coins": int(a["gt_coins"]),
                "detect_approx": f"{_rate(a['found'], a['gt_coins']):.3f}",
                "classify_approx": f"{_rate(a['typed'], a['found']):.3f}",
                "total_match_rate": f"{_rate(a['total_ok'], a['images']):.3f}",
                "count_exact_rate": f"{_rate(a['count_ok'], a['images']):.3f}",
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
    parser = argparse.ArgumentParser(description="이미지 단위 정답으로 하는 근사 평가")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["tune", "test", "ext"])
    parser.add_argument("--out", type=Path, default=Path("eval/results/summary_counts.csv"))
    args = parser.parse_args()

    rows = evaluate(
        load_counts(args.labels),
        load_manifest(args.manifest),
        load_pred_values(args.preds),
        set(args.splits) if args.splits else None,
    )
    if not rows:
        raise SystemExit("평가할 데이터가 없습니다 (split 또는 파일 이름을 확인하세요)")

    print("※ 위치 정답이 없어 개수 기반 근사치입니다.")
    for r in rows:
        print(
            f"[{r['split']}/{r['layout']}] 이미지 {r['images']}장 | "
            f"검출(근사) {r['detect_approx']} | 분류(근사) {r['classify_approx']} | "
            f"총금액 {r['total_match_rate']} | 개수 완전일치 {r['count_exact_rate']}"
        )
    write_summary(rows, args.out)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
