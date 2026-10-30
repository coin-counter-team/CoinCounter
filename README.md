# CoinCounter

> 사진 속 동전을 영상처리로 검출하고 종류별 개수와 총금액을 자동으로 계산하는 Python/OpenCV 프로젝트 (심화전공실습 팀 프로젝트)

## 주요 기능

| 기능 | 설명 | 담당 |
| --- | --- | --- |
| 전처리 | Grayscale, Gaussian Blur, 이진화, 형태학적 연산, 밝기·대비 조절, 히스토그램 평활화 | 유승화 |
| 동전 검출 | Contour 기반 / Hough Circle 기반 검출, 면적·원형도로 오검출 제거 | 오현우 |
| 특징 추출·분류 | 크기(반지름), HSV 색상, ORB 특징점 매칭 기반 분류 규칙 | 정휘 |
| 개수·금액 계산 | 종류별 개수와 총금액 계산 | 정휘 |
| 평가·CNN 비교 | 검출률·분류 정확도·총금액 일치율 측정, CNN 분류기와 비교 | 정준영 |

## 대상 동전과 촬영 조건

| 항목 | 내용 |
| --- | --- |
| 대상 동전 | 10원(신형), 50원, 100원, 500원 |
| 기본 조건 | 단색 배경, 고정된 촬영 거리, 수직 촬영, 동전끼리 겹치지 않음 |
| 확장 조건 | 조명 3종(기본·밝음·어두움) × 배경 2종(밝은 단색·어두운 단색) |

## 처리 흐름

```
입력 이미지 → 전처리(preprocess) → 동전 검출(detect) → 특징 추출(features)
→ 분류(classify) → 개수·금액 계산(count) → 결과 표시(main)
```

## 폴더 구조

| 경로 | 내용 |
| --- | --- |
| `src/coin/` | 처리 단계별 모듈 (`preprocess`, `detect`, `features`, `classify`, `count`, `main`) |
| `tests/` | pytest 테스트 |
| `data/` | 촬영 가이드와 정답(labels) — 원본 이미지는 저장소에 올리지 않음 |
| `eval/` | 평가 스크립트와 조건별 평가표 |
| `docs/` | 보고서·발표 자료·회의록 등 문서 |
| `.github/` | 이슈·PR 템플릿, CI 워크플로 |

## 실행 방법

Python 3.11 이상을 권장합니다.

**Windows (PowerShell)**

```powershell
git clone https://github.com/coin-counter-team/CoinCounter.git
cd CoinCounter
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m coin.main data\raw\sample.jpg
pytest
```

**macOS / Linux**

```bash
git clone https://github.com/coin-counter-team/CoinCounter.git
cd CoinCounter
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m coin.main data/raw/sample.jpg
pytest
```

> 현재는 뼈대 단계로, `count.total_amount` 외 함수는 아직 구현되지 않았습니다.

## 목표 지표

| 데이터 | 규모 | 검출률 | 분류 정확도 | 총금액 일치율 |
| --- | --- | --- | --- | --- |
| 기본 조건 | 100장 이상 (조정용 60 / 테스트 40) | 85% 이상 | 85% 이상 | 85% 이상 |
| 확장 조건 (조명 3 × 배경 2) | 조건별 10장, 60장 이상 | 80% 이상 | 80% 이상 | 80% 이상 |
| CNN 분류기 | 동일한 테스트 동전 영역 | — | 85% 이상 | — |

## 일정

| 단계 | 기간 |
| --- | --- |
| 데이터 수집 | 10.27 ~ 11.02 |
| 데이터 전처리 | 11.03 ~ 11.09 |
| 구현 | 11.10 ~ 11.23 |
| 중간 점검·확장 조건 | 11.24 ~ 11.30 |
| 최종 평가·발표 준비 | 12.01 ~ 12.09 |

## 팀원과 역할

| 이름 | 역할 | 주요 담당 |
| --- | --- | --- |
| 유승화 | 팀장 | 일정 관리, 촬영 조건·데이터 구성, 정답 기록 관리, 전처리, 모듈 통합 |
| 오현우 | 검출 | Contour·Hough Circle 검출, 오검출 제거, 두 방식 비교 |
| 정휘 | 특징·분류 | 크기·HSV 특징, ORB 매칭, 분류 규칙, 개수·총금액 계산 |
| 정준영 | 평가·문서 | 지표 측정, 조건별 결과 정리, CNN 비교, 보고서·발표 |

모든 팀원이 데이터 촬영과 정답 기록에 참여합니다.

## 기여

GitHub Flow를 따릅니다. `main`에서 `feature/<작업명>` 브랜치를 만들어 작업하고 PR로 병합합니다. 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

## AI 사용 고지

생성형 AI(Claude)를 저장소 구성·코드 작성 보조에 사용하며, 팀원이 직접 검증·수정함.

## 라이선스

[MIT](LICENSE)
