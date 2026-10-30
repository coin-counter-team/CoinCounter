"""특징 추출 모듈 (담당: 정휘).

동전 영역의 크기(반지름), HSV 색상, ORB 특징점을 추출한다.
"""

import numpy as np

from coin.detect import Circle


def crop_coin(image: np.ndarray, circle: Circle) -> np.ndarray:
    """원 영역만 잘라낸 동전 이미지를 반환한다."""
    raise NotImplementedError


def hsv_features(coin_bgr: np.ndarray) -> np.ndarray:
    """동전 영역의 HSV 색상 통계(평균 등)를 반환한다."""
    raise NotImplementedError


def orb_match_score(coin_gray: np.ndarray, reference_gray: np.ndarray) -> float:
    """기준 이미지와의 ORB 특징점 매칭 점수를 반환한다."""
    raise NotImplementedError
