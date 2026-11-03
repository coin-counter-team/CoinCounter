import cv2
import numpy as np
import pytest

from coin.hybrid import refine_with_contours


@pytest.mark.parametrize("foreground,background", [(220, 30), (30, 220)])
def test_contour_refines_radius_and_center(foreground, background):
    image = np.full((260, 260), background, dtype=np.uint8)
    cv2.circle(image, (134, 127), 55, foreground, -1)
    result = refine_with_contours(image, [(130, 130, 50)])[0]
    assert result.reason == "contour"
    assert result.contour is not None
    assert abs(result.circle[0] - 134) <= 1
    assert abs(result.circle[1] - 127) <= 1
    assert abs(result.circle[2] - 55) <= 2


def test_missing_contour_preserves_original():
    candidates = [(130, 130, 50), (10, 10, 20), (100, 100, 3)]
    results = refine_with_contours(np.zeros((260, 260), dtype=np.uint8), candidates)
    assert [r.circle for r in results] == candidates
    assert all(r.contour is None for r in results)
    assert [r.reason for r in results] == ["no_valid_contour", "image_border", "small_radius"]


def test_rectangle_cannot_replace_hough_circle():
    image = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(image, (85, 125), (215, 175), 220, -1)
    original = (150, 150, 50)
    result = refine_with_contours(image, [original])[0]
    assert result.circle == original
    assert result.contour is None


def test_neighbor_is_not_used_as_target_contour():
    image = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(image, (205, 150), 45, 220, -1)
    original = (130, 150, 50)
    result = refine_with_contours(image, [original])[0]
    assert result.circle == original
    assert result.contour is None


def test_empty_candidates_and_invalid_image():
    assert refine_with_contours(np.zeros((100, 100), dtype=np.uint8), []) == []
    with pytest.raises(ValueError):
        refine_with_contours(np.zeros((100, 100, 3), dtype=np.uint8), [])
