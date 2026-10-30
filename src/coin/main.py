"""실행 진입점 (통합: 유승화).

사용법: PYTHONPATH=src python -m coin.main <이미지 경로>
"""

import argparse


def run(image_path: str) -> dict[int, int]:
    """이미지 한 장을 처리해 종류별 개수를 반환한다."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="동전 종류 인식 및 총금액 계산")
    parser.add_argument("image", help="입력 이미지 경로")
    args = parser.parse_args()
    run(args.image)


if __name__ == "__main__":
    main()
