"""실행 진입점 (통합: 유승화).

사용법: PYTHONPATH=src python -m coin.main <이미지 경로>

터미널 실행: main() → process_image() → format_result()로 결과 출력.
다른 Python 코드에서는 run()으로 종류별 개수만 얻거나,
process_image()로 미분류 개수까지 함께 얻을 수 있다.
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
    """counts는 {금액: 개수}, unknown은 금액을 판단하지 못해 합계에서 제외한 개수."""

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

    px_per_mm는 1mm당 픽셀 수이며, 생략하면 분류기가 동전 크기·색상으로 추정한다.
    배율을 추정할 정보가 부족한 사진은 분류기의 기본 배율을 사용한다.
    Hough 검출은 이진 마스크 대신 원본의 회색조 영상을 사용한다.
    """
    # Python에서 직접 호출하는 경우에도 잘못된 설정을 처리 전에 걸러낸다.
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

    # Contour는 전처리된 이진 마스크, Hough는 회색조 사진에서 원을 찾는다.
    # circles의 각 원소는 (중심 x, 중심 y, 반지름)이며 단위는 픽셀이다.
    if detector == "contour":
        circles = detect_contour(preprocess_image(image, method))
    else:
        circles = detect_hough(to_gray(image))
    # 분류에는 원본의 색상이 필요하다. 결과 중 None은 미분류로 따로 세고,
    # 금액이 확인된 동전만 종류별로 집계한다.
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
    """터미널 입력을 처리하고 인식 결과 또는 오류 메시지를 출력한다."""
    # 옵션을 생략하면 Contour + adaptive 전처리 + 사진별 배율 추정을 사용한다.
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
        help="1mm당 픽셀 수. 생략하면 사진 속 동전 크기·색상으로 추정",
    )
    args = parser.parse_args()
    try:
        result = process_image(
            args.image, detector=args.detector, method=args.method, px_per_mm=args.px_per_mm
        )
    except (OSError, ValueError, cv2.error) as error:
        # 파일·설정·영상 처리 오류는 짧은 메시지와 실패 종료 코드(1)로 알린다.
        parser.exit(1, f"오류: {error}\n")
    # format_result가 종류별 개수로 총금액을 계산하고 미분류 안내를 붙인다.
    print(format_result(result.counts, result.unknown))


if __name__ == "__main__":
    main()
