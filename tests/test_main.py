"""파일 입력부터 검출·분류·출력까지 이어지는 통합 테스트."""

import sys

import cv2
import numpy as np
import pytest

from coin.main import main, process_image, run


@pytest.fixture
def mixed_photo(tmp_path):
    image = np.full((300, 600, 3), 245, dtype=np.uint8)
    for x, radius, color in (
        (70, 36, (60, 110, 200)),
        (200, 43, (150, 150, 150)),
        (340, 48, (150, 150, 150)),
        (490, 53, (150, 150, 150)),
    ):
        cv2.circle(image, (x, 150), radius, color, -1)
    path = tmp_path / "동전 사진.png"
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)
    return path


@pytest.mark.parametrize("method", ["adaptive", "otsu"])
def test_contour_pipeline_counts_each_denomination(mixed_photo, method):
    result = process_image(mixed_photo, method=method)
    assert result.counts == {10: 1, 50: 1, 100: 1, 500: 1}
    assert result.unknown == 0
    assert run(mixed_photo, method=method) == result.counts


def test_hough_pipeline_counts_each_denomination(mixed_photo):
    result = process_image(mixed_photo, detector="hough", px_per_mm=4.0)
    assert result.counts == {10: 1, 50: 1, 100: 1, 500: 1}
    assert result.unknown == 0


def test_cli_reports_unknown_and_excludes_it_from_total(tmp_path, monkeypatch, capsys):
    image = np.full((300, 400, 3), 245, dtype=np.uint8)
    cv2.circle(image, (90, 150), 48, (150, 150, 150), -1)  # 100원
    cv2.circle(image, (270, 150), 70, (150, 150, 150), -1)  # 규격 밖
    path = tmp_path / "unknown.png"
    cv2.imencode(".png", image)[1].tofile(path)
    monkeypatch.setattr(sys, "argv", ["coin.main", str(path), "--px-per-mm", "4"])
    main()
    assert capsys.readouterr().out.strip() == "100원 1개 / 총 100원 (미분류 1개, 금액에서 제외)"


def test_empty_scene(tmp_path):
    path = tmp_path / "empty.png"
    cv2.imencode(".png", np.full((200, 200, 3), 245, dtype=np.uint8))[1].tofile(path)
    result = process_image(path)
    assert result.counts == {10: 0, 50: 0, 100: 0, 500: 0}
    assert result.unknown == 0


@pytest.mark.parametrize("content", [b"", b"not an image"])
def test_unreadable_image(tmp_path, content):
    path = tmp_path / "broken.png"
    path.write_bytes(content)
    with pytest.raises(OSError, match="사진을 읽을 수 없습니다"):
        run(path)


def test_cli_missing_file_exits_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["coin.main", str(tmp_path / "missing.png")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "오류:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_rejected_even_without_coins(tmp_path, scale):
    with pytest.raises(ValueError, match="유한한 양수"):
        process_image(tmp_path / "unused.png", px_per_mm=scale)


@pytest.mark.parametrize("options", [{"detector": "invalid"}, {"method": "invalid"}])
def test_invalid_pipeline_options(tmp_path, options):
    with pytest.raises(ValueError, match="지원하지 않는"):
        process_image(tmp_path / "unused.png", **options)
