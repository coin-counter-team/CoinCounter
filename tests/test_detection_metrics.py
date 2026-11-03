from coin.detection_metrics import detection_counts


def test_duplicate_is_false_positive():
    assert detection_counts([(10, 10, 20), (11, 10, 20)], [(10, 10, 20)]) == (1, 1, 0)


def test_matching_reassigns_ambiguous_candidate():
    # 첫 후보는 두 정답 모두에 가깝고, 둘째 후보는 첫 정답에만 가깝다.
    assert detection_counts([(5, 0, 10), (0, 0, 10)], [(0, 0, 10), (10, 0, 10)]) == (2, 0, 0)


def test_empty_and_missing():
    assert detection_counts([], []) == (0, 0, 0)
    assert detection_counts([], [(0, 0, 10)]) == (0, 0, 1)
    assert detection_counts([(100, 100, 10)], [(0, 0, 10)]) == (0, 1, 1)
