import cv2
import numpy as np

from coin.detect import detect_contour


def test_detect_contour_finds_two_circles():
    mask = np.zeros((300, 500), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 30, 255, -1)
    cv2.circle(mask, (280, 180), 45, 255, -1)

    assert sorted(detect_contour(mask)) == [
        (100, 100, 30),
        (280, 180, 45),
    ]


def test_detect_contour_empty_image():
    mask = np.zeros((300, 500), dtype=np.uint8)

    assert detect_contour(mask) == []


def test_detect_contour_removes_small_noise():
    mask = np.zeros((300, 500), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 30, 255, -1)
    cv2.circle(mask, (200, 50), 3, 255, -1)

    assert detect_contour(mask) == [(100, 100, 30)]


def test_detect_contour_removes_long_rectangle():
    mask = np.zeros((300, 500), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 30, 255, -1)
    cv2.rectangle(mask, (370, 40), (390, 180), 255, -1)

    assert detect_contour(mask) == [(100, 100, 30)]
