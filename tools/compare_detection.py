"""두 검출 방식의 원·이미지·개수 비교를 저장한다 (전처리는 비교용 임시 구현)."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from coin.detect import detect_contour, detect_hough
from coin.detection_metrics import detection_counts
from coin.hybrid import refine_with_contours


def prepare(image: np.ndarray, max_side: int, background: str):
    """같은 축소 영상에서 비교한다. 실제 가로·세로 배율을 반환한다."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if background == "light" else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, 0, 255, mode | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return small, gray, mask, (w / small.shape[1], h / small.shape[0])


def load_counts(path: str | None) -> dict[str, int]:
    if path is None:
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # 촬영 개수표와 저장소 표준의 동전 한 개당 한 줄인 CSV를 지원한다.
    if rows and "파일" in rows[0]:
        return {r["파일"]: int(r["동전 수"]) for r in rows}
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["image"]] = counts.get(row["image"], 0) + 1
    return counts


def load_truth(path: str | None) -> dict[str, list[tuple[float, float, float]]]:
    truth: dict[str, list[tuple[float, float, float]]] = {}
    if path:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if {"image", "x", "y", "r"}.issubset(reader.fieldnames or []):
                for row in reader:
                    truth.setdefault(row["image"], []).append(
                        tuple(float(row[k]) for k in ("x", "y", "r"))
                    )
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--labels", help="hw_labels.csv 또는 표준 coins.csv")
    parser.add_argument("--output", type=Path, default=Path("eval/results/detection"))
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--background", choices=("light", "dark"), default="light")
    parser.add_argument("--min-area", type=float, default=300)
    parser.add_argument("--min-circularity", type=float, default=0.75)
    parser.add_argument("--min-radius", type=int, default=20)
    parser.add_argument("--max-radius", type=int, default=120)
    parser.add_argument("--min-dist", type=float, default=50)
    parser.add_argument("--param1", type=float, default=100)
    parser.add_argument("--param2", type=float, default=30)
    parser.add_argument("--center-ratio", type=float, default=0.5)
    parser.add_argument("--hybrid", action="store_true", help="Hough+Contour 보정을 세 번째에 표시")
    args = parser.parse_args()
    if args.max_side < 1:
        parser.error("max-side는 양수여야 합니다")
    paths = sorted(args.image_dir.glob(args.pattern))
    if not paths:
        parser.error("입력 사진이 없습니다")
    counts = load_counts(args.labels)
    truth = load_truth(args.labels)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "settings.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary, detections, metrics, refinements, timings = [], [], [], [], []
    for path in paths:
        # Windows 한글 경로도 읽을 수 있도록 imdecode를 사용한다.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
        small, gray, mask, (sx, sy) = prepare(image, args.max_side, args.background)
        started = perf_counter()
        contour_circles = detect_contour(mask, args.min_area, args.min_circularity)
        contour_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        hough_circles = detect_hough(
            gray,
            min_dist=args.min_dist,
            param1=args.param1,
            param2=args.param2,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
        )
        hough_ms = (perf_counter() - started) * 1000
        methods = {"contour": contour_circles, "hough": hough_circles}
        timings.extend([[path.name, "contour", contour_ms], [path.name, "hough", hough_ms]])
        raw_hough = methods["hough"]
        hybrid_results = []
        if args.hybrid:
            started = perf_counter()
            hybrid_results = refine_with_contours(gray, raw_hough)
            hybrid_ms = hough_ms + (perf_counter() - started) * 1000
            timings.append([path.name, "hybrid", hybrid_ms])
            methods["hybrid"] = [r.circle for r in hybrid_results]
            for i, result in enumerate(hybrid_results, 1):
                ox, oy, old_r = result.original
                nx, ny, new_r = result.circle
                refinements.append(
                    [
                        path.name,
                        i,
                        result.reason,
                        round(ox * sx),
                        round(oy * sy),
                        round(old_r * (sx + sy) / 2),
                        round(nx * sx),
                        round(ny * sy),
                        round(new_r * (sx + sy) / 2),
                    ]
                )
        panels = []
        for method, circles in methods.items():
            original = [
                (round(x * sx), round(y * sy), round(r * (sx + sy) / 2)) for x, y, r in circles
            ]
            if path.name in truth:
                tp, fp, fn = detection_counts(original, truth[path.name], args.center_ratio)
                metrics.append(
                    [
                        path.name,
                        method,
                        tp,
                        fp,
                        fn,
                        tp / (tp + fn) if tp + fn else "",
                        tp / (tp + fp) if tp + fp else "",
                    ]
                )
            panel = small.copy()
            for index, (x, y, r) in enumerate(circles, 1):
                color = (0, 255, 0)
                if method == "hybrid":
                    selected = hybrid_results[index - 1]
                    if selected.contour is None:
                        color = (0, 165, 255)
                    else:
                        cv2.drawContours(panel, [selected.contour], -1, (0, 255, 255), 1)
                cv2.circle(panel, (x, y), r, color, 2)
                cv2.circle(panel, (x, y), 3, (0, 0, 255), -1)
                detections.append(
                    [
                        path.name,
                        method,
                        index,
                        round(x * sx),
                        round(y * sy),
                        round(r * (sx + sy) / 2),
                    ]
                )
            cv2.putText(
                panel,
                f"{method}: {len(circles)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            panels.append(panel)
            if method == "hybrid":
                changed = sum(r.contour is not None for r in hybrid_results)
                cv2.putText(
                    panel,
                    f"contour {changed}, keep {len(circles) - changed}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            method_dir = args.output / method
            method_dir.mkdir(exist_ok=True)
            ok, separate = cv2.imencode(".jpg", panel)
            if not ok:
                raise RuntimeError(f"개별 결과 저장 실패: {path}")
            separate.tofile(method_dir / f"{path.stem}_{method}.jpg")
            expected = counts.get(path.name)
            summary.append(
                [
                    path.name,
                    method,
                    len(circles),
                    "" if expected is None else expected,
                    "" if expected is None else len(circles) == expected,
                ]
            )
        ok, encoded = cv2.imencode(".jpg", np.hstack(panels))
        if not ok:
            raise RuntimeError(f"이미지 저장 실패: {path}")
        encoded.tofile(args.output / f"{path.stem}_comparison.jpg")
    for filename, header, rows in (
        ("counts.csv", ["image", "method", "detected", "expected", "count_match"], summary),
        ("circles.csv", ["image", "method", "index", "x", "y", "r"], detections),
        ("metrics.csv", ["image", "method", "tp", "fp", "fn", "recall", "precision"], metrics),
        (
            "refinements.csv",
            ["image", "index", "reason", "old_x", "old_y", "old_r", "new_x", "new_y", "new_r"],
            refinements,
        ),
        ("timings.csv", ["image", "method", "detection_ms"], timings),
    ):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    for method in methods:
        rows = [r for r in summary if r[1] == method and r[3] != ""]
        print(f"{method}: {sum(r[4] for r in rows)}/{len(rows)} images count match")
    print("개수 일치는 위치 기반 검출률이 아닙니다. 결과 이미지를 함께 확인하세요.")


if __name__ == "__main__":
    main()
