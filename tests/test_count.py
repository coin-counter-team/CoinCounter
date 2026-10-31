import pytest

from coin.count import count_coins, format_result, split_unknown, total_amount


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


def test_count_coins_empty():
    assert count_coins([]) == {10: 0, 50: 0, 100: 0, 500: 0}


def test_count_coins_rejects_unknown_value():
    with pytest.raises(ValueError):
        count_coins([7])


def test_split_unknown():
    assert split_unknown([10, None, 500, None]) == ([10, 500], 2)


def test_format_result():
    assert format_result({10: 2, 50: 0, 100: 0, 500: 1}) == "10원 2개, 500원 1개 / 총 520원"
    assert format_result(count_coins([])) == "동전 없음 / 총 0원"
    assert format_result({100: 1}, unknown=2) == "100원 1개 / 총 100원 (미분류 2개, 금액에서 제외)"
