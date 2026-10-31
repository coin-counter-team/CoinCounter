"""특징 추출 모듈 (담당: 정휘).

동전 영역의 크기(반지름), HSV 색상, ORB 특징점을 추출한다.
"""

import cv2
import numpy as np

from coin.detect import Circle

# HSV 평균을 낼 때 동전 반지름의 몇 %까지만 쓸지.
# 가장자리는 배경·그림자가 섞이기 쉬워서 안쪽 80%만 사용한다.
HSV_MASK_RATIO = 0.8


def crop_coin(image: np.ndarray, circle: Circle) -> np.ndarray:
    """원 영역만 잘라낸 동전 이미지를 반환한다.

    원을 감싸는 정사각형을 잘라내고, 원 바깥(모서리) 픽셀은 0(검정)으로 채운다.
    이미지 경계를 벗어나는 부분은 잘린다.
    """
    x, y, r = (round(v) for v in circle)
    h, w = image.shape[:2]
    x0, y0 = max(x - r, 0), max(y - r, 0)
    x1, y1 = min(x + r + 1, w), min(y + r + 1, h)
    crop = image[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x - x0, y - y0), r, 255, -1)
    crop[mask == 0] = 0
    return crop


def radius_mm(circle: Circle, px_per_mm: float) -> float:
    """픽셀 반지름을 실제 크기(mm)로 바꾼다. px_per_mm은 1mm가 몇 픽셀인지."""
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm은 0보다 커야 함: {px_per_mm}")
    return circle[2] / px_per_mm


def hsv_features(coin_bgr: np.ndarray) -> np.ndarray:
    """동전 영역의 HSV 색상 통계(평균 등)를 반환한다.

    crop_coin 결과(정사각형, 가운데 원)를 받아 안쪽 원(HSV_MASK_RATIO) 안의
    평균 [H, S, V]를 반환한다. OpenCV 기준 H는 0~179, S·V는 0~255.
    """
    h, w = coin_bgr.shape[:2]
    hsv = cv2.cvtColor(coin_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), dtype=np.uint8)
    r = max(int(min(h, w) / 2 * HSV_MASK_RATIO), 1)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    mean = cv2.mean(hsv, mask=mask)[:3]
    return np.array(mean, dtype=np.float64)


# ORB 설정.
# nfeatures=500은 OpenCV cv2.ORB_create의 기본값과 같다
# (https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html).
ORB_N_FEATURES = 500
# 좋은 매칭으로 인정할 최대 해밍 거리 (0~256 비트 중).
# 64는 근거 없는 추측값이다. 실제 사진의 거리 분포를 재서 정할 것.
ORB_MAX_DISTANCE = 64


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def orb_match_score(coin: np.ndarray, reference: np.ndarray) -> float:
    """기준 이미지와의 ORB 특징점 매칭 점수를 반환한다. 컬러/흑백 모두 가능.
    1) 두 이미지에서 ORB 특징점·디스크립터를 뽑고
    2) BFMatcher(NORM_HAMMING, crossCheck=True)로 서로 가장 가까운 쌍만 매칭한 뒤
    3) 해밍 거리가 ORB_MAX_DISTANCE 이하인 '좋은 매칭' 수를
       두 이미지 중 적은 특징점 수로 나눈 비율(0~1)을 점수로 쓴다.
    특징점이 하나도 없으면 0.0.
    """
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    _, des1 = orb.detectAndCompute(_to_gray(coin), None)
    _, des2 = orb.detectAndCompute(_to_gray(reference), None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    good = [m for m in matches if m.distance <= ORB_MAX_DISTANCE]
    return len(good) / min(len(des1), len(des2))


def classify_by_orb(coin: np.ndarray, templates: dict[int, np.ndarray]) -> tuple[int | None, float]:
    """ORB 점수가 가장 높은 템플릿의 금액과 그 점수를 반환한다.

    templates는 {금액: 기준 동전 crop 이미지}. 크기·색상 규칙과 비교하기 위한
    보조 분류기이며, 템플릿이 없거나 모든 점수가 0이면 (None, 0.0).
    """
    best_value: int | None = None
    best_score = 0.0
    for value, template in templates.items():
        score = orb_match_score(coin, template)
        if score > best_score:
            best_value, best_score = value, score
    return best_value, best_score
