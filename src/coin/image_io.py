"""Windows 한글 경로를 포함한 이미지 읽기."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """BGR uint8 영상을 읽는다. 누락·빈 파일·손상 영상은 명시적으로 실패한다."""
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error) as exc:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}") from exc
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image
