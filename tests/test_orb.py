import cv2
import numpy as np

from coin.features import classify_by_orb, orb_match_score


def _coin(text: str, size: int = 200) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), c - 5, 180, -1)
    cv2.circle(img, (c, c), c - 15, 60, 3)
    cv2.putText(img, text, (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4)
    return img


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h))


def test_identical_scores_higher_than_different():
    a, b = _coin("100"), _coin("XO#")
    assert orb_match_score(a, a) > orb_match_score(a, b)


def test_rotated_scores_higher_than_different():
    a, b = _coin("100"), _coin("XO#")
    assert orb_match_score(_rotate(a, 30), a) > orb_match_score(b, a)


def test_blank_image_returns_zero():
    blank = np.zeros((100, 100), dtype=np.uint8)
    assert orb_match_score(blank, _coin("100")) == 0.0


def test_classify_by_orb_picks_matching_template():
    templates = {100: _coin("100"), 500: _coin("XO#")}
    value, score = classify_by_orb(_rotate(_coin("100"), 20), templates)
    assert value == 100 and score > 0
    assert classify_by_orb(_coin("100"), {}) == (None, 0.0)
