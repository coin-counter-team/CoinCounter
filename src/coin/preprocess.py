"""전처리 모듈 (담당: 유승화).

Grayscale, Gaussian Blur, 이진화, 형태학적 연산과
조명 대응(밝기·대비 조절, 히스토그램 평활화)을 담당한다.
"""

import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 Grayscale로 변환한다."""
    raise NotImplementedError


def enhance(gray: np.ndarray, method: str = "equalize") -> np.ndarray:
    """밝기·대비 조절 또는 히스토그램 평활화를 적용한다."""
    raise NotImplementedError


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Blur 후 이진화하고 형태학적 연산으로 노이즈를 줄인 마스크를 반환한다."""
    raise NotImplementedError
