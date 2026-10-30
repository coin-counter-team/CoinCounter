"""분류 모듈 (담당: 정휘, CNN 비교: 정준영).

크기·HSV·ORB 특징을 이용한 규칙 기반 분류로 동전 금액(10/50/100/500)을 정한다.
"""

import numpy as np

from coin.detect import Circle


def classify_coin(image: np.ndarray, circle: Circle) -> int:
    """동전 하나의 금액을 반환한다."""
    raise NotImplementedError


def classify_all(image: np.ndarray, circles: list[Circle]) -> list[int]:
    """검출된 모든 동전의 금액 목록을 반환한다."""
    raise NotImplementedError
