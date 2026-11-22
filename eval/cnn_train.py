"""작은 CNN 학습과 규칙 기반 비교 (담당: 정준영).

cnn_data.py가 만든 동전 그림(.npz)으로 작은 CNN을 학습한다.
  - 학습: split == tune 인 동전만 (test는 절대 학습에 쓰지 않는다)
  - 평가: split == test 인 동전. 규칙 기반 분류기 예측(rule 열)과 같은 동전에서 비교한다.
test 동전이 없으면 평가할 수 없어 tune 정확도(학습에 쓴 데이터라 참고용)만 보여 준다.

실행 (저장소 루트에서, PyTorch 필요):
    python eval/cnn_train.py --data eval/results/cnn_crops.npz --epochs 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CLASSES = (10, 50, 100, 500)


def make_model(num_classes: int = len(CLASSES)) -> nn.Module:
    def block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 16),
        block(16, 32),
        block(32, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float 0~1"""
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0


def augment(batch: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """동전은 방향이 없으므로 회전·뒤집기와 밝기·대비 변화를 준다. 색 자체는 바꾸지 않는다."""
    out = []
    for img in batch:
        img = torch.rot90(img, int(torch.randint(0, 4, (1,), generator=gen)), dims=(1, 2))
        if torch.rand(1, generator=gen) < 0.5:
            img = torch.flip(img, dims=(2,))
        contrast = 0.8 + 0.4 * torch.rand(1, generator=gen)
        brightness = -0.1 + 0.2 * torch.rand(1, generator=gen)
        mean = img.mean()
        out.append(((img - mean) * contrast + mean + brightness).clamp(0, 1))
    return torch.stack(out)


def train(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_model()
    counts = torch.bincount(y, minlength=len(CLASSES)).float().clamp(min=1)
    weights = counts.sum() / (len(CLASSES) * counts)  # 적은 종류에 더 큰 가중치
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_size = 16
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:  # BatchNorm은 1개짜리 배치를 못 쓴다
                continue
            loss = loss_fn(model(augment(x[idx], gen)), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {total / len(x):.4f}")
    return model


def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).argmax(dim=1).numpy()


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean()) if len(truth) else 0.0


def confusion(pred: np.ndarray, truth: np.ndarray) -> str:
    """행 = 정답, 열 = 예측 (CLASSES 순서)"""
    lines = ["정답\\예측  " + " ".join(f"{c:>4d}" for c in CLASSES)]
    for t, c in enumerate(CLASSES):
        row = [int(((truth == t) & (pred == p)).sum()) for p in range(len(CLASSES))]
        lines.append(f"{c:>8d}  " + " ".join(f"{n:>4d}" for n in row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="작은 CNN 학습과 규칙 기반 비교")
    parser.add_argument("--data", type=Path, default=ROOT / "eval/results/cnn_crops.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/cnn_compare.csv")
    args = parser.parse_args()

    data = np.load(args.data)
    to_class = {c: i for i, c in enumerate(CLASSES)}
    y_all = np.array([to_class[int(v)] for v in data["y"]])
    rule_all = np.array([to_class.get(int(v), -1) for v in data["rule"]])  # -1 = 분류 실패
    split = data["split"]

    train_mask = split == "tune"
    if not train_mask.any():
        raise SystemExit("학습할 tune 동전이 없습니다")
    x_all = to_tensor(data["x"])
    model = train(
        x_all[train_mask], torch.from_numpy(y_all[train_mask]), args.epochs, args.lr, args.seed
    )

    rows = []
    for name in ("tune", "test"):
        mask = split == name
        if not mask.any():
            print(f"[{name}] 동전이 없어 평가를 건너뜁니다")
            continue
        truth = y_all[mask]
        cnn_pred = predict(model, x_all[mask])
        rule_pred = rule_all[mask]
        note = "학습에 쓴 데이터(참고용)" if name == "tune" else "평가"
        print(f"\n[{name}] 동전 {int(mask.sum())}개 - {note}")
        print(f"  CNN       정확도 {accuracy(cnn_pred, truth):.3f}")
        print(f"  규칙 기반 정확도 {accuracy(rule_pred, truth):.3f} (분류 실패는 오답)")
        print(
            "  CNN 혼동 행렬\n"
            + "\n".join("    " + s for s in confusion(cnn_pred, truth).split("\n"))
        )
        rows.append(
            {
                "split": name,
                "coins": int(mask.sum()),
                "cnn_accuracy": f"{accuracy(cnn_pred, truth):.3f}",
                "rule_accuracy": f"{accuracy(rule_pred, truth):.3f}",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "coins", "cnn_accuracy", "rule_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n저장: {args.out}")
    print("※ 동전 수가 적어 소규모 예비 비교입니다. 한 개 차이가 정확도를 크게 바꿉니다.")


if __name__ == "__main__":
    main()
