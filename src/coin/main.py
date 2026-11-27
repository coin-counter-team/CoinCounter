"""실행 진입점 (통합: 유승화).

사용법: PYTHONPATH=src python -m coin.main <이미지 경로>
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coin.classify import classify_all
from coin.count import count_coins, format_result, split_unknown
from coin.detect import detect_contour, detect_hough
from coin.preprocess import preprocess_image, to_gray


@dataclass
class CoinResult:
    """종류별 개수와 총금액에서 제외한 미분류 동전 수."""

    counts: dict[int, int]
    unknown: int


def process_image(
    image_path: str | Path,
    *,
    detector: str = "contour",
    method: str = "adaptive",
    px_per_mm: float | None = None,
) -> CoinResult:
    """사진을 읽어 검출·분류하고 개수와 미분류 수를 반환한다.

    px_per_mm를 생략하면 기존 분류기의 사진별 배율 추정을 사용한다.
    Hough 검출은 이진 마스크 대신 원본의 회색조 영상을 사용한다.
    """
    if detector not in {"contour", "hough"}:
        raise ValueError(f"지원하지 않는 검출 방식: {detector}")
    if method not in {"adaptive", "otsu"}:
        raise ValueError(f"지원하지 않는 이진화 방식: {method}")
    if px_per_mm is not None and (not np.isfinite(px_per_mm) or px_per_mm <= 0):
        raise ValueError("px_per_mm은 유한한 양수여야 합니다")

    # imread 대신 바이트를 디코딩해 Windows의 한글 경로도 지원한다.
    raw = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if image is None:
        raise OSError(f"사진을 읽을 수 없습니다: {image_path}")

    if detector == "contour":
        circles = detect_contour(preprocess_image(image, method))
    else:
        circles = detect_hough(to_gray(image))
    known, unknown = split_unknown(classify_all(image, circles, px_per_mm))
    return CoinResult(count_coins(known), unknown)


def run(
    image_path: str | Path,
    *,
    detector: str = "contour",
    method: str = "adaptive",
    px_per_mm: float | None = None,
) -> dict[int, int]:
    """이미지 한 장을 처리해 종류별 개수를 반환한다. 미분류는 개수에서 제외한다."""
    return process_image(image_path, detector=detector, method=method, px_per_mm=px_per_mm).counts


def main() -> None:
    parser = argparse.ArgumentParser(description="동전 종류 인식 및 총금액 계산")
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--detector", choices=("contour", "hough"), default="contour")
    parser.add_argument(
        "--method",
        choices=("adaptive", "otsu"),
        default="adaptive",
        help="Contour 검출의 이진화 방식 (기본: adaptive)",
    )
    parser.add_argument(
        "--px-per-mm",
        type=float,
        help="1mm당 픽셀 수. 생략하면 사진 속 10원으로 추정하고, 없으면 기본값 사용",
    )
    args = parser.parse_args()
    try:
        result = process_image(
            args.image, detector=args.detector, method=args.method, px_per_mm=args.px_per_mm
        )
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"오류: {error}\n")
    print(format_result(result.counts, result.unknown))


if __name__ == "__main__":
    main()
