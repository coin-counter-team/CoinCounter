# Hough + Contour 결합 검출

Hough가 찾은 동전 후보 주변에서 Contour를 검증해 중심과 반지름을 보정한다.
적절한 윤곽선을 찾지 못하면 원래 Hough 원을 그대로 유지한다.
Hough 구현 브랜치 `feature/hough-detect`를 기반으로 한 추가 기능이다.
기존의 별도 반지름 보정 실험 코드는 포함하지 않는다.

## 처리 흐름

1. 원본 회색조 영상에서 Hough로 후보를 찾는다.
2. 후보 반지름의 약 1.45배보다 넓은 영역을 잘라낸다.
3. Gaussian Blur → Canny → 작은 틈 연결 → Contour 검출을 적용한다.
4. 면적·원형도·중심 거리·반지름 범위·원 맞춤 오차·방향 분포를 검사한다.
5. 조건에 맞는 윤곽선에 원을 맞춰 좌표·반지름을 보정한다.
6. 실패하면 Hough 결과를 유지한다. 후보의 개수와 순서는 유지한다.

초록 원을 그린 사진을 다시 분석하는 것이 아니라, 원을 그리기 전의 회색조를 사용한다.

```python
from coin.detect import detect_hough
from coin.hybrid import refine_with_contours

candidates = detect_hough(gray, min_dist=90, min_radius=45, max_radius=110, param1=150, param2=50)
details = refine_with_contours(gray, candidates)
circles = [item.circle for item in details]
```

설정 예시는 긴 변 1200픽셀로 축소한 촬영 사진 기준이다.
입력은 비어 있지 않은 uint8 회색조 영상이며 출력 좌표는 입력 해상도 기준이다.
`item.original`은 원래 Hough 원, `item.circle`은 최종 원,
`item.contour`는 선택한 윤곽선 (없으면 None), `item.reason`은 적용·유지 이유다.
사진 가장자리여서 영역 확보가 어렵거나 너무 작은 후보는 보정하지 않는다.

## 비교 도구

의존성을 설치한 환경에서 저장소 루트를 기준으로 실행한다.
사진은 `data/raw`에, 정답은 동전당 한 줄인 표준 `data/labels/coins.csv`에 둔다.
개수만 있는 `hw_labels.csv`도 지원한다. 정답 없이도 실행할 수 있다.

```powershell
$env:PYTHONPATH = "src"
python tools/compare_detection.py data/raw --labels data/labels/coins.csv --output eval/results/hybrid --min-radius 45 --max-radius 110 --min-dist 90 --param1 150 --param2 50 --hybrid
```

- `*_comparison.jpg`: 왼쪽 Contour / 가운데 Hough / 오른쪽 Hybrid.
- `contour/`, `hough/`, `hybrid/`: 방식별 개별 결과 이미지.
- Hybrid의 초록 원: Contour 채택, 주황 원: Hough 유지, 노란 선: 선택한 윤곽선.
- `counts.csv`: 개수 비교. 일치해도 위치가 모두 맞다는 뜻은 아니다.
- `circles.csv`: 원본 해상도로 환산한 좌표·반지름.
- `refinements.csv`: 원본 해상도 기준 보정 전후 원과 채택/유지 이유.
- `metrics.csv`: 좌표 정답이 있을 때만 일대일 중심 매칭으로 TP·FP·FN·검출률·정밀도 기록.
- `timings.csv`: 읽기·공통 전처리·그리기·저장 제외 검출 시간. Hybrid는 Hough+보정 합계.
- `settings.json`: 재현용 설정.

비교 도구의 Contour 전처리는 Otsu 이진화·닫힘 연산의 임시 구현이다.
`--background light/dark`로 배경에 맞는 극성을 선택한다.
거리는 축소 영상 픽셀, 면적은 축소 영상 픽셀 제곱 기준이다.
split은 자동 분리하지 않으므로 평가할 split별 폴더와 CSV를 따로 준비한다.
정답 CSV에 행이 없는 빈 이미지는 평가 대상으로 표현할 수 없다는 제한이 있다.
중심 매칭은 정답 반지름의 0.5배 이내를 기본으로 한다 (`--center-ratio`).
정답과 예측을 최대 개수의 일대일 매칭으로 연결하고 중복 예측은 오검출로 센다.
검출률은 TP/(TP+FN), 정밀도는 TP/(TP+FP)이며 분모가 0이면 빈칸이다.
중심 매칭은 반지름 정확도를 측정하지 않는다.

## 제한과 평가

- Hough가 놓친 동전을 새로 찾거나 잘못된 후보를 제거하는 기능은 아니다.
- 내부 테두리나 그림자가 조건을 통과할 수 있어 항상 개선된다고 보장하지 않는다.
- 로컬 사진 22장은 설정 조정에도 사용했으므로 최종 테스트 성능으로 보고하지 않는다.
- 크기 분류 개선 여부는 별도의 정답 중심·반지름을 측정해 검증해야 한다.
- 원본 사진과 자동 생성 결과는 업로드하지 않는다.
- 테스트는 명확한 원의 보정과 실패 시 원본 유지, 사각형·이웃 동전 배제를 확인한다.
