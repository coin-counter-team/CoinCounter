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
from coin.features import crop_coin, hsv_features, radius_mm

# 한국은행 공시 동전 지름(mm). 10원은 2006년 이후 신형(구리 도금 알루미늄).
COIN_DIAMETER_MM: dict[int, float] = {10: 18.0, 50: 21.6, 100: 24.0, 500: 26.5}

# 1mm가 몇 픽셀인지. 촬영 거리를 고정하므로 상수로 둔다.
# 실제 사진에서 동전(또는 자) 하나를 재서 "반지름 px / 반지름 mm"로 다시 정할 것.
PX_PER_MM: float = 4.0

# 구리색(10원) 판정 기준. OpenCV HSV에서 H는 0~179(빨강 0, 주황 ~10~20, 노랑 ~30).
# 구리는 주황 계열이고 채도가 높다. 은색(50/100/500원)은 채도가 낮아 S가 작다.
COPPER_HUE_RANGE: tuple[float, float] = (0.0, 25.0)
COPPER_MIN_SATURATION: float = 80.0
# 가장 가까운 동전 지름과 이만큼(mm) 넘게 다르면 '모르겠음'(None).
MAX_SIZE_DIFF_MM: float = 1.5


def is_copper(hsv_mean: np.ndarray) -> bool:
    """HSV 평균이 구리색(10원 신형)인지 판단한다."""
    h, s = float(hsv_mean[0]), float(hsv_mean[1])
    lo, hi = COPPER_HUE_RANGE
    return lo <= h <= hi and s >= COPPER_MIN_SATURATION


def classify_by_size(diameter_mm: float, candidates=(50, 100, 500)) -> int | None:
    """지름(mm)이 가장 가까운 동전 금액. 차이가 MAX_SIZE_DIFF_MM 초과면 None."""
    best = min(candidates, key=lambda v: abs(COIN_DIAMETER_MM[v] - diameter_mm))
    if abs(COIN_DIAMETER_MM[best] - diameter_mm) > MAX_SIZE_DIFF_MM:
        return None
    return best


def classify_coin(image: np.ndarray, circle: Circle, px_per_mm: float = PX_PER_MM) -> int | None:
    """동전 하나의 금액을 반환한다. 판단할 수 없으면 None."""
    if is_copper(hsv_features(crop_coin(image, circle))):
        return 10
    return classify_by_size(2 * radius_mm(circle, px_per_mm))


def classify_all(
    image: np.ndarray, circles: list[Circle], px_per_mm: float = PX_PER_MM
) -> list[int | None]:
    """검출된 모든 동전의 금액 목록을 반환한다. 분류 실패는 None."""
    return [classify_coin(image, c, px_per_mm) for c in circles]
