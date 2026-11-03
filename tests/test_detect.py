import cv2
import numpy as np
import pytest

from coin.detect import detect_contour, detect_hough


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


def test_hough_empty_scene():
    assert detect_hough(np.zeros((200, 200), dtype=np.uint8)) == []


def test_hough_finds_circles_with_tolerance():
    gray = np.zeros((300, 500), dtype=np.uint8)
    expected = [(100, 100, 30), (280, 180, 45)]
    for x, y, r in expected:
        cv2.circle(gray, (x, y), r, 255, -1)
    found = detect_hough(gray, min_dist=80, min_radius=25, max_radius=50, param2=20)
    assert len(found) == len(expected)
    for x, y, r in expected:
        assert any(
            abs(cx - x) <= 3 and abs(cy - y) <= 3 and abs(cr - r) <= 3 for cx, cy, cr in found
        )


def test_hough_excludes_out_of_range_circle():
    gray = np.zeros((300, 500), dtype=np.uint8)
    cv2.circle(gray, (100, 100), 10, 255, -1)
    cv2.circle(gray, (280, 180), 45, 255, -1)
    found = detect_hough(gray, min_radius=35, max_radius=55, param2=20)
    assert len(found) == 1
    assert abs(found[0][0] - 280) <= 3
    assert abs(found[0][2] - 45) <= 3


@pytest.mark.parametrize(
    "gray",
    [
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.zeros((20, 20), dtype=float),
        np.zeros((0, 0), dtype=np.uint8),
    ],
)
def test_hough_rejects_invalid_image(gray):
    with pytest.raises(ValueError):
        detect_hough(gray)


@pytest.mark.parametrize(
    "options",
    [
        {"blur_size": 4},
        {"dp": 0},
        {"param2": -1},
        {"min_dist": float("nan")},
        {"min_radius": 50, "max_radius": 20},
    ],
)
def test_hough_rejects_invalid_settings(options):
    with pytest.raises(ValueError):
        detect_hough(np.zeros((100, 100), dtype=np.uint8), **options)
