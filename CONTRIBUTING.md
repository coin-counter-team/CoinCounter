# 기여 가이드

## 작업 흐름 (GitHub Flow)

1. 작업 전에 이슈를 만들거나 배정된 이슈를 확인합니다.
2. 최신 `main`에서 브랜치를 만듭니다.
   ```bash
   git switch main && git pull
   git switch -c feature/hough-detect
   ```
3. 작은 단위로 커밋하고 push 합니다.
4. PR을 열고 템플릿을 채운 뒤 `Closes #이슈번호`로 이슈를 연결합니다.
5. 팀원 1명 이상의 리뷰와 CI 통과 후 `main`에 병합합니다.

## 브랜치 이름

| 접두사 | 용도 | 예시 |
| --- | --- | --- |
| `feature/` | 새 기능 | `feature/contour-detect` |
| `fix/` | 버그 수정 | `fix/hsv-threshold` |
| `docs/` | 문서 | `docs/eval-report` |
| `chore/` | 설정·기타 | `chore/ci-cache` |

## 커밋 메시지

`<타입>: <한국어 요약>` 형식을 사용합니다.

| 타입 | 의미 | 예시 |
| --- | --- | --- |
| `feat` | 기능 추가 | `feat: Hough Circle 검출 추가` |
| `fix` | 버그 수정 | `fix: 500원 반지름 기준 수정` |
| `docs` | 문서 | `docs: 촬영 가이드 보완` |
| `chore` | 설정·빌드·기타 | `chore: ruff 설정 변경` |
| `test` | 테스트 | `test: 금액 계산 테스트 추가` |

## 코드 스타일과 테스트

```bash
pip install -r requirements.txt ruff
ruff check .
ruff format .
pytest
```

PR 전에 위 명령이 모두 통과해야 합니다.

## 데이터 규칙

- 원본 사진(`data/raw/`)은 저장소에 올리지 않고 팀 공유 드라이브에 보관합니다.
- 정답 CSV(`data/labels/`)만 저장소에 올립니다. 형식은 [data/README.md](data/README.md)를 따릅니다.
- 비슷한 연속 사진은 같은 묶음(`group`)으로 관리해 조정용과 테스트에 동시에 들어가지 않게 합니다.
- 개인정보(얼굴, 연락처 등)가 담긴 파일은 올리지 않습니다.

## AI 사용

생성형 AI를 사용한 경우 PR 본문에 사용 범위를 적고, 결과는 직접 검증한 뒤 올립니다.
