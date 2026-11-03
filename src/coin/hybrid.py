"""Hough 후보 주변의 Contour를 검증하고, 실패하면 Hough 원을 유지한다."""

from dataclasses import dataclass

import cv2
import numpy as np

from coin.detect import Circle


@dataclass
class Refinement:
    original: Circle
    circle: Circle
    contour: np.ndarray | None
    reason: str


def refine_with_contours(gray: np.ndarray, candidates: list[Circle]) -> list[Refinement]:
    """원본 회색조에서 후보별 윤곽선을 찾는다. 후보 개수·순서는 유지한다.

    경계 검출 후 작은 틈만 연결한다. 면적, 원형도, 중심 거리, 반지름 범위,
    원 맞춤 오차와 방향 분포로 검증한다. 이 조건은 정확도 보장이 아니며
    내부 테두리나 그림자를 선택할 수 있어 실제 정답으로 평가해야 한다.
    반환 contour 좌표는 입력 영상 기준이며, 실패 시 None이다.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("비어 있지 않은 uint8 회색조 이미지가 필요합니다")
    h, w = gray.shape
    results = []
    for original in candidates:
        x, y, r = original
        fallback = Refinement(original, original, None, "no_valid_contour")
        if r < 5:
            fallback.reason = "small_radius"
            results.append(fallback)
            continue
        margin = int(np.ceil(r * 1.45)) + 3
        x0, y0 = x - margin, y - margin
        x1, y1 = x + margin + 1, y + margin + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            fallback.reason = "image_border"
            results.append(fallback)
            continue
        roi = gray[y0:y1, x0:x1]
        smooth = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(smooth, 50, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        best_score = -np.inf
        best = fallback
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if len(contour) < 12 or perimeter <= 0 or area < np.pi * (r * 0.8) ** 2:
                continue
            circularity = 4 * np.pi * area / perimeter**2
            if circularity < 0.7:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= 1 or by <= 1 or bx + bw >= roi.shape[1] - 1:
                continue
            if by + bh >= roi.shape[0] - 1:
                continue
            points = contour[:, 0, :].astype(float) + (x0 - x, y0 - y)
            solution, _, rank, _ = np.linalg.lstsq(
                np.column_stack((2 * points, np.ones(len(points)))),
                np.sum(points**2, axis=1),
                rcond=None,
            )
            cx, cy, constant = solution
            radius = float(np.sqrt(max(0, constant + cx * cx + cy * cy)))
            shift = np.hypot(cx, cy)
            if rank < 3 or shift > r * 0.2 or not r * 0.8 <= radius <= r * 1.3:
                continue
            centered = points - (cx, cy)
            residual = np.sqrt(np.mean((np.linalg.norm(centered, axis=1) - radius) ** 2))
            if residual > max(1.5, radius * 0.06):
                continue
            sectors = (
                (np.arctan2(centered[:, 1], centered[:, 0]) + np.pi) * 8 / (2 * np.pi)
            ).astype(int) % 8
            if len(np.unique(sectors)) < 8:
                continue
            # 비슷하게 원형인 후보 중 바깥 경계를 우선하되 중심 이동·오차에 감점.
            score = radius / r + circularity - shift / r - residual / radius
            if score > best_score:
                best_score = score
                absolute = contour.copy()
                absolute[:, 0, :] += (x0, y0)
                best = Refinement(
                    original, (round(x + cx), round(y + cy), round(radius)), absolute, "contour"
                )
        results.append(best)
    return results
