# Hough Circle 동전 검출

담당: 오현우. `src/coin/detect.py`의 `detect_hough`에 구현했다.
기존 Contour·면적·원형도 필터는 변경하지 않았다.

## 입출력

- 입력: 비어 있지 않은 uint8 2차원 회색조 영상.
- 출력: 입력 영상 픽셀 기준 `(x, y, r)` 정수 목록. 검출이 없으면 `[]`.
- 잘못된 영상과 주요 설정값에는 `ValueError`를 발생시킨다.
- Gaussian Blur 후 `cv2.HoughCircles`를 적용한다.

```python
from coin.detect import detect_hough

circles = detect_hough(gray)
# 긴 변 1200픽셀로 축소한 로컬 촬영 사진의 탐색 설정 예시:
circles = detect_hough(gray, min_dist=90, min_radius=45,
                       max_radius=110, param1=150, param2=50)
```

위 설정은 다른 촬영 거리·해상도에 그대로 적용할 수 없다.
`min_dist`는 중심 간 최소 거리로 중복 검출을 억제하지만 너무 크면 가까운 동전을 놓친다.
`min_radius`·`max_radius`는 허용 크기, `param1`은 내부 경계 검출 기준,
`param2`는 원 후보 채택 기준이다. `max_radius=0`은 OpenCV의 자동 상한을 사용한다.
`dp`는 중심 누적 영상의 해상도 비율, `blur_size`는 블러 커널의 양의 홀수 크기다.

## 제한과 검증

Hough 원이 내부 문양이나 그림자를 선택할 수 있으므로 반지름 정확도를 보장하지 않는다.
촬영 조건에 맞게 조정하고 별도의 테스트 사진으로 누락·오검출·반지름 오차를 확인해야 한다.
원 검출 허용오차, 원이 없는 영상, 반지름 제한과 잘못된 입력에 대한 테스트를 포함한다.
이 변경에는 Hybrid나 추가 반지름 보정, 비교 도구를 포함하지 않는다.

저장소 루트에서 의존성을 설치한 환경으로 검사한다.

```powershell
pytest
ruff check .
ruff format --check src/coin/detect.py tests/test_detect.py
```
