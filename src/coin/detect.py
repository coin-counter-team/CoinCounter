"""동전 검출 모듈 (담당: 오현우).

Contour 기반과 Hough Circle 기반 검출을 구현하고,
면적·원형도로 오검출을 제거한다. 결과는 (x, y, r) 원 목록이다.
"""

import cv2
import numpy as np

Circle = tuple[int, int, int]


def detect_contour(
    mask: np.ndarray,
    min_area: float = 100.0,
    min_circularity: float = 0.75,
) -> list[Circle]:
    """이진 마스크에서 면적·원형도 조건에 맞는 동전 후보를 찾는다."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return filter_candidates(contours, min_area, min_circularity)


def detect_hough(gray: np.ndarray) -> list[Circle]:
    """Grayscale 이미지에서 Hough Circle로 동전을 찾는다."""
    raise NotImplementedError


def filter_candidates(
    contours: list[np.ndarray],
    min_area: float,
    min_circularity: float,
) -> list[Circle]:
    """윤곽선의 면적·원형도로 오검출을 제거하고 원 목록을 반환한다."""
    circles: list[Circle] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 1:
            continue

        circles.append((round(x), round(y), round(radius)))

    return circles
