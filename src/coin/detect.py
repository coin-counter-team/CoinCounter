"""동전 검출 모듈 (담당: 오현우).

Contour 기반과 Hough Circle 기반 검출을 구현하고,
면적·원형도로 오검출을 제거한다. 결과는 (x, y, r) 원 목록이다.
"""

import numpy as np

Circle = tuple[int, int, int]


def detect_contour(mask: np.ndarray) -> list[Circle]:
    """이진 마스크에서 윤곽선으로 동전 후보를 찾는다."""
    raise NotImplementedError


def detect_hough(gray: np.ndarray) -> list[Circle]:
    """Grayscale 이미지에서 Hough Circle로 동전을 찾는다."""
    raise NotImplementedError


def filter_candidates(
    circles: list[Circle], min_area: float, min_circularity: float
) -> list[Circle]:
    """면적·원형도 기준에 맞지 않는 후보를 제거한다."""
    raise NotImplementedError
