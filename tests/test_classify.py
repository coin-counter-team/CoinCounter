"""합성 이미지(cv2.circle)로 만든 분류 테스트. 실제 사진 기준값은 별도로 조정해야 함."""

import cv2
import numpy as np
import pytest

from coin.classify import (
    COIN_DIAMETER_MM,
    PX_PER_MM,
    classify_all,
    classify_by_size,
    classify_coin,
    copper_ratio,
    is_copper,
)
from coin.features import crop_coin, hsv_features, radius_mm

COPPER_BGR = (60, 110, 200)  # 주황빛 구리색
SILVER_BGR = (190, 190, 195)  # 채도가 낮은 은색
BACKGROUND_BGR = (40, 40, 40)


def make_image(coins):
    """coins: [(x, y, r, bgr)] 를 그린 합성 이미지."""
    img = np.full((400, 600, 3), BACKGROUND_BGR, dtype=np.uint8)
    for x, y, r, color in coins:
        cv2.circle(img, (x, y), r, color, -1)
    return img


def r_px(value):
    return round(COIN_DIAMETER_MM[value] / 2 * PX_PER_MM)


def test_crop_coin_masks_outside():
    img = make_image([(100, 100, 30, SILVER_BGR)])
    crop = crop_coin(img, (100, 100, 30))
    assert crop.shape[:2] == (61, 61)
    assert crop[0, 0].tolist() == [0, 0, 0]
    assert crop[30, 30].tolist() == list(SILVER_BGR)


def test_crop_coin_clips_at_border():
    img = make_image([(5, 5, 20, SILVER_BGR)])
    assert crop_coin(img, (5, 5, 20)).shape[:2] == (26, 26)


def test_radius_mm():
    assert radius_mm((0, 0, 40), 4.0) == 10.0
    with pytest.raises(ValueError):
        radius_mm((0, 0, 40), 0)


def test_hsv_copper_vs_silver():
    copper = crop_coin(make_image([(100, 100, 40, COPPER_BGR)]), (100, 100, 40))
    silver = crop_coin(make_image([(100, 100, 40, SILVER_BGR)]), (100, 100, 40))
    assert is_copper(copper)
    assert not is_copper(silver)


@pytest.mark.parametrize("value", [50, 100, 500])
def test_classify_silver_by_size(value):
    r = r_px(value)
    img = make_image([(200, 200, r, SILVER_BGR)])
    assert classify_coin(img, (200, 200, r)) == value


def test_classify_copper_is_10():
    r = r_px(10)
    img = make_image([(200, 200, r, COPPER_BGR)])
    assert classify_coin(img, (200, 200, r)) == 10


def test_classify_all_mixed():
    coins = [
        (80, 100, r_px(10), COPPER_BGR),
        (200, 100, r_px(50), SILVER_BGR),
        (320, 100, r_px(100), SILVER_BGR),
        (460, 100, r_px(500), SILVER_BGR),
    ]
    img = make_image(coins)
    circles = [(x, y, r) for x, y, r, _ in coins]
    assert classify_all(img, circles) == [10, 50, 100, 500]


def test_copper_ratio_handles_red_hue_wrap():
    """H가 0 근처와 179 근처로 반반 섞여도(평균은 ~90) 구리로 판정돼야 한다."""
    img = np.zeros((41, 41, 3), dtype=np.uint8)
    img[:, :20] = (40, 60, 200)  # H ≈ 3
    img[:, 20:] = (60, 40, 200)  # H ≈ 176
    assert hsv_features(img)[0] > 60  # 평균만 쓰면 엉뚱한 값
    assert copper_ratio(img) > 0.9
    assert is_copper(img)


def test_crop_coin_rounds_float_circle():
    """HoughCircles가 준 실수 좌표도 반올림해서 정상적으로 잘라야 한다 (F3)."""
    img = make_image([(100, 100, 30, SILVER_BGR)])
    assert crop_coin(img, (100.4, 99.6, 30.2)).shape[:2] == (61, 61)


def test_hsv_edge_coin_same_as_center():
    """이미지 가장자리에서 잘린 동전도 가운데 동전과 같게 구리로 판정해야 한다 (F4)."""
    center = crop_coin(make_image([(200, 100, 36, COPPER_BGR)]), (200, 100, 36))
    edge = crop_coin(make_image([(5, 100, 36, COPPER_BGR)]), (5, 100, 36))
    assert is_copper(center)
    assert is_copper(edge)
    assert abs(copper_ratio(edge) - copper_ratio(center)) < 0.05


def test_classify_by_size_boundary():
    """허용 오차(1.5mm) 안이면 가장 가까운 동전으로 분류한다 (C4)."""
    assert classify_by_size(20.2) == 50
    assert classify_by_size(20.0) is None


def test_classify_by_size_unknown():
    """어느 동전과도 크기가 크게 다르면 None (C4)."""
    assert classify_by_size(35.0) is None
