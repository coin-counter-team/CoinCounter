import pytest

from coin.count import count_coins, total_amount


def test_total_amount_mixed():
    assert total_amount({10: 3, 50: 1, 100: 2, 500: 1}) == 780


def test_total_amount_empty():
    assert total_amount({}) == 0


def test_total_amount_rejects_unknown_value():
    with pytest.raises(ValueError):
        total_amount({5: 1})


def test_total_amount_rejects_negative():
    with pytest.raises(ValueError):
        total_amount({100: -1})


def test_count_coins():
    assert count_coins([10, 500, 10]) == {10: 2, 50: 0, 100: 0, 500: 1}
