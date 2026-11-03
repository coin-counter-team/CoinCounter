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


def detect_hough(
    gray: np.ndarray,
    *,
    dp: float = 1.2,
    min_dist: float = 40.0,
    param1: float = 100.0,
    param2: float = 30.0,
    min_radius: int = 10,
    max_radius: int = 0,
    blur_size: int = 5,
) -> list[Circle]:
    """8비트 회색조 이미지에서 원을 찾는다. 좌표·반지름 단위는 입력 픽셀.

    기본값은 시작점이며 촬영 해상도에 맞춰 조정한다. max_radius=0은 상한 자동.
    min_dist로 같은 동전의 중복 검출을 억제한다.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("비어 있지 않은 uint8 회색조 이미지가 필요합니다")
    if not all(np.isfinite(v) and v > 0 for v in (dp, min_dist, param1, param2)):
        raise ValueError("dp, min_dist, param1, param2는 양수여야 합니다")
    if min_radius < 0 or max_radius < 0 or (max_radius and max_radius <= min_radius):
        raise ValueError("반지름은 0 이상이며 상한은 하한보다 커야 합니다")
    if blur_size < 1 or blur_size % 2 == 0:
        raise ValueError("blur_size는 양의 홀수여야 합니다")

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [tuple(int(v) for v in np.rint(circle)) for circle in circles[0]]


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
