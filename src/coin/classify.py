"""분류 모듈 (담당: 정휘, CNN 비교: 정준영).

크기·HSV 특징을 이용한 규칙 기반 분류로 동전 금액(10/50/100/500)을 정한다.

규칙:
1. 색상: 평균 색이 구리색(주황 계열 + 채도 높음)이면 10원(신형).
2. 크기: 아니면 실제 지름(mm)이 가장 가까운 은색 동전(50/100/500원)으로 정한다.

주의: 아래 임계값은 tools/measure_coins.py로 실제 사진(조정용 데이터)에서 다시 맞춘다.
ORB 매칭(#7)은 features.py에 구현돼 있지만 이 규칙에는 아직 쓰지 않는다
(비교 계획은 docs/orb-matching.md)."""

import numpy as np

from coin.detect import Circle
from coin.features import crop_coin, hue_histogram, radius_mm

# 한국은행 공시 동전 지름(mm). 10원은 2006년 이후 신형(구리 도금 알루미늄).
COIN_DIAMETER_MM: dict[int, float] = {10: 18.0, 50: 21.6, 100: 24.0, 500: 26.5}

# 1mm가 몇 픽셀인지. 사진마다 촬영 높이가 달라 estimate_px_per_mm()로 사진별로 구하고,
# 사진에 10원이 없을 때만 이 값을 쓴다. 단일 동전 사진에서 잰 시작값(약 21)이다.
PX_PER_MM: float = 21.0

# 구리색(10원) 판정 기준. OpenCV HSV에서 H는 0~179(빨강 0·179, 주황 ~10~20, 노랑 ~30).
# Hue는 원형이라 빨강 쪽(170~179)도 구리 범위로 본다. 은색 동전은 채도가 낮아 S가 작다.
# 실제 사진에서 은색 동전도 H 19~26으로 나와서 상한을 25 → 18, 채도 기준을 80 → 100으로 바꿨다.
COPPER_HUE_RANGES: tuple[tuple[int, int], ...] = ((0, 18), (170, 179))
COPPER_MIN_SATURATION: float = 100.0
# 동전 픽셀 중 구리색 픽셀이 이 비율 이상이면 10원.
COPPER_MIN_RATIO: float = 0.5
# 가장 가까운 동전 지름과 이만큼(mm) 넘게 다르면 '모르겠음'(None).
MAX_SIZE_DIFF_MM: float = 1.5
# 구리색이어도 이보다 크면 10원이 아니다(10원 18mm와 50원 21.6mm의 중간).
COPPER_MAX_DIAMETER_MM: float = (COIN_DIAMETER_MM[10] + COIN_DIAMETER_MM[50]) / 2


def copper_ratio(coin_bgr: np.ndarray) -> float:
    """동전 픽셀 중 구리색(H 범위 + 채도 기준) 픽셀의 비율(0~1)."""
    hist = hue_histogram(coin_bgr, COPPER_MIN_SATURATION)
    return float(sum(hist[lo : hi + 1].sum() for lo, hi in COPPER_HUE_RANGES))


def is_copper(coin_bgr: np.ndarray) -> bool:
    """crop된 동전 이미지가 구리색(10원 신형)인지 판단한다."""
    return copper_ratio(coin_bgr) >= COPPER_MIN_RATIO


def classify_by_size(diameter_mm: float, candidates=(50, 100, 500)) -> int | None:
    """지름(mm)이 가장 가까운 동전 금액. 차이가 MAX_SIZE_DIFF_MM 초과면 None."""
    best = min(candidates, key=lambda v: abs(COIN_DIAMETER_MM[v] - diameter_mm))
    if abs(COIN_DIAMETER_MM[best] - diameter_mm) > MAX_SIZE_DIFF_MM:
        return None
    return best


def estimate_px_per_mm(
    image: np.ndarray, circles: list[Circle], default: float = PX_PER_MM
) -> float:
    """사진 속 10원(구리색) 동전의 지름으로 이 사진의 px/mm를 구한다.

    10원 지름(18mm)을 자처럼 쓴다. 10원이 없으면 default를 쓴다.
    구리색으로 잘못 잡힌 큰 동전을 빼려고 가장 작은 구리색 동전의 1.1배 이하만 쓴다.
    """
    diameters = sorted(2 * c[2] for c in circles if is_copper(crop_coin(image, c)))
    if not diameters:
        return default
    reference = [d for d in diameters if d <= diameters[0] * 1.1]
    return float(np.median(reference)) / COIN_DIAMETER_MM[10]


def classify_coin(image: np.ndarray, circle: Circle, px_per_mm: float = PX_PER_MM) -> int | None:
    """동전 하나의 금액을 반환한다. 판단할 수 없으면 None."""
    diameter = 2 * radius_mm(circle, px_per_mm)
    if is_copper(crop_coin(image, circle)) and diameter < COPPER_MAX_DIAMETER_MM:
        return 10
    return classify_by_size(diameter)


def classify_all(
    image: np.ndarray, circles: list[Circle], px_per_mm: float | None = None
) -> list[int | None]:
    """검출된 모든 동전의 금액 목록을 반환한다. 분류 실패는 None.

    px_per_mm를 주지 않으면 estimate_px_per_mm()로 사진마다 구한다.
    """
    if px_per_mm is None:
        px_per_mm = estimate_px_per_mm(image, circles)
    return [classify_coin(image, c, px_per_mm) for c in circles]
