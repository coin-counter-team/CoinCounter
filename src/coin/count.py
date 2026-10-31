"""개수·금액 계산 모듈 (담당: 정휘)."""

from collections import Counter
from collections.abc import Iterable

from coin import COIN_VALUES


def split_unknown(values: Iterable[int | None]) -> tuple[list[int], int]:
    """분류 결과에서 None(분류 실패)을 떼어 (금액 목록, 미분류 개수)로 나눈다."""
    known: list[int] = []
    unknown = 0
    for v in values:
        if v is None:
            unknown += 1
        else:
            known.append(v)
    return known, unknown


def count_coins(values: Iterable[int]) -> dict[int, int]:
    """분류된 금액 목록을 종류별 개수로 센다. None은 split_unknown으로 먼저 걸러낼 것."""
    counts = {v: 0 for v in COIN_VALUES}
    for value, n in Counter(values).items():
        if value not in counts:
            raise ValueError(f"지원하지 않는 동전 금액: {value}")
        counts[value] = n
    return counts


def total_amount(counts: dict[int, int]) -> int:
    """종류별 개수 {금액: 개수}로 총금액을 계산한다."""
    total = 0
    for value, n in counts.items():
        if value not in COIN_VALUES:
            raise ValueError(f"지원하지 않는 동전 금액: {value}")
        if n < 0:
            raise ValueError(f"개수는 0 이상이어야 함: {value}원 {n}개")
        total += value * n
    return total


def format_result(counts: dict[int, int], unknown: int = 0) -> str:
    """'10원 2개, 500원 1개 / 총 520원' 형식의 결과 문자열. 0개인 종류는 생략."""
    parts = [f"{v}원 {n}개" for v, n in sorted(counts.items()) if n > 0]
    text = (", ".join(parts) or "동전 없음") + f" / 총 {total_amount(counts)}원"
    if unknown:
        text += f" (미분류 {unknown}개, 금액에서 제외)"
    return text
