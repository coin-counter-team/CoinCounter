# 평가

| 지표 | 정의 |
| --- | --- |
| 검출률 | 정답 동전 중 올바르게 검출된 비율 (중심 거리 기준 매칭) |
| 분류 정확도 | 올바르게 검출된 동전 중 금액을 맞힌 비율 |
| 총금액 일치율 | 계산한 총금액이 정답과 정확히 일치한 이미지 비율 |

조건별(기본, 배경 2 × 조명 3) 결과와 Contour/Hough, 특징 기반/CNN 비교표를 이 폴더에 정리합니다. 실행 결과물은 `eval/results/`에 저장되며 저장소에는 올리지 않습니다.

## 정답 형식별 평가 스크립트

| 스크립트 | 정답 | 비고 |
| --- | --- | --- |
| `evaluate.py` | 동전 한 개당 한 줄(`x, y, r, value`) | 위 정의 그대로 계산 |
| `evaluate_counts.py` | 사진 한 장당 한 줄(종류별 개수, 총액) | 위치 정답이 없는 동안 사용 |

`evaluate_counts.py`는 검출 방식을 비교할 때 쓰도록 `detect_count_match`(예측 동전 수가 정답과 같은 사진의 비율)와 `pred_per_gt`(예측 수 ÷ 정답 수)도 출력한다. `detect_approx`는 엉뚱한 원이 많아도 1에 가까울 수 있어서, 과다 검출(예: Hough 기본값) 여부는 이 두 값으로 본다.

`evaluate_counts.py`의 총금액 일치율과 개수 완전일치율은 정확한 값이다. 검출률과 분류 정확도는 위치를 모르므로 개수만 비교한 **근사치**이며, 엉뚱한 곳을 잡은 원은 반영하지 못한다. 보고서에는 근사치임을 밝힌다.

예측 CSV는 `make_preds.py`로 만든다. `--labels`에는 `image`, `split` 열이 있는 CSV(예: `image_manifest.csv`)를 넘긴다.

## CNN 비교 (예비)

동전별 정답이 없어서, 정답이 곧 동전별 정답이 되는 `single`·`same` 사진의 동전만 사용한다. 학습은 `tune`, 평가는 `test` 동전으로만 하며, 같은 동전에서 규칙 기반 분류와 비교한다.

```bash
PYTHONPATH=src python eval/cnn_data.py --labels <coins.csv> --manifest <image_manifest.csv> \
    --images data/raw --splits tune test
python eval/cnn_train.py --data eval/results/cnn_crops.npz --epochs 60 --seed 0
```

`cnn_train.py`는 PyTorch가 필요하며 `requirements.txt`에는 넣지 않았다. 동전이 학습 27개·평가 23개뿐이라 시드에 따라 결과가 흔들리므로 시드를 바꿔 여러 번 돌려 평균을 본다. 소규모 예비 비교로만 해석한다.
