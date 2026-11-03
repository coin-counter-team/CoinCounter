"""좌표 정답이 있을 때 검출 원을 일대일로 매칭한다."""

import math

from coin.detect import Circle


def detection_counts(
    predicted: list[Circle], truth: list[Circle], center_ratio: float = 0.5
) -> tuple[int, int, int]:
    """(TP, FP, FN). 중심 거리가 정답 반지름×비율 이내면 매칭 후보.

    최대 개수의 일대일 매칭으로 중복 검출을 정답으로 여러 번 세지 않는다.
    반지름 오차는 이 지표에 포함하지 않는다. 비율은 평가 전에 고정한다.
    """
    if not math.isfinite(center_ratio) or center_ratio <= 0:
        raise ValueError("center_ratio는 양수여야 합니다")
    edges = [
        [
            j
            for j, (tx, ty, tr) in enumerate(truth)
            if math.hypot(px - tx, py - ty) <= tr * center_ratio
        ]
        for px, py, _ in predicted
    ]
    assigned: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    tp = sum(augment(i, set()) for i in range(len(predicted)))
    return tp, len(predicted) - tp, len(truth) - tp
