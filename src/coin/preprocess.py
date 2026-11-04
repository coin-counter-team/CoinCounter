"""전처리 모듈 (담당: 유승화).

Grayscale, Gaussian Blur, 이진화, 형태학적 연산과
조명 대응(밝기·대비 조절, 히스토그램 평활화)을 담당한다.
"""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 Grayscale로 변환한다."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance(gray: np.ndarray, method: str = "equalize") -> np.ndarray:
    """밝기·대비 조절 또는 히스토그램 평활화를 적용한다."""
    if method == "equalize":
        return cv2.equalizeHist(gray)
    if method == "stretch":
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    raise ValueError(f"지원하지 않는 명암 보정 방식: {method}")


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Blur 후 이진화하고 형태학적 연산으로 노이즈를 줄인 마스크를 반환한다."""
    short_side = min(gray.shape)
    scale = max(1, round(short_side / 600))
    blur_size = 4 * scale + 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    if method == "otsu":
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 테두리의 다수 픽셀을 배경으로 보고 배경을 검은색으로 맞춘다.
        border = np.concatenate((mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]))
        if np.count_nonzero(border) > border.size / 2:
            mask = cv2.bitwise_not(mask)
    elif method == "adaptive":
        # 밝은 배경에서 주변 평균보다 어두운 영역을 전경으로 잡는다.
        window = max(3, round(short_side / 4) | 1)
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, window, 12
        )
    else:
        raise ValueError(f"지원하지 않는 이진화 방식: {method}")

    for operation, radius in ((cv2.MORPH_OPEN, scale), (cv2.MORPH_CLOSE, 2 * scale)):
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, operation, kernel)

    # 바깥 배경과 연결되지 않은 동전 내부의 구멍을 채운다.
    background = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(background, None, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(background[1:-1, 1:-1]))


def preprocess_image(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """BGR 사진을 원본 크기의 검출용 이진 마스크로 변환한다."""
    gray = to_gray(image)
    mask = binarize(gray, method)
    if method == "adaptive":
        mask = _repair_coin_masks(gray, mask)
        mask = _clean_coin_mask(mask)
    return mask


def _clean_coin_mask(mask: np.ndarray) -> np.ndarray:
    """복원 후 남은 작은 잡점을 제거하고 동전 내부의 빈 픽셀을 채운다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(1, mask.size * 0.00015)
    cleaned = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(cleaned, [contour], -1, 255, cv2.FILLED)
    return cleaned


def _coin_rims(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """회색조 영상에서 여러 방향의 테두리가 확인되는 원만 반환한다."""
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    circles = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT, 1, 12, param1=75, param2=22, minRadius=17, maxRadius=64
    )
    if circles is None:
        return []
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    scored = []
    for x, y, radius in circles[0]:
        if min(x, y, gray.shape[1] - 1 - x, gray.shape[0] - 1 - y) < radius + 4:
            continue
        samples = radius + np.arange(-3, 4)[:, None]
        xs = np.rint(x + samples * cos).astype(int)
        ys = np.rint(y + samples * sin).astype(int)
        dx, dy = gx[ys, xs], gy[ys, xs]
        radial = np.abs(dx * cos + dy * sin)
        edges = (radial > 28) & (radial >= 0.8 * np.hypot(dx, dy))
        sectors = edges.any(axis=0).reshape(8, -1).mean(axis=1)
        if sectors.mean() >= 0.88 and sectors.min() >= 0.5:
            scored.append((sectors.mean(), float(x), float(y), float(radius)))

    rims = []
    for _, x, y, radius in sorted(scored, reverse=True):
        if all(np.hypot(x - cx, y - cy) >= 0.85 * (radius + r) for cx, cy, r in rims):
            rims.append((x, y, radius))
    return rims


def _repair_coin_masks(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """원래 마스크가 붙거나 깨진 곳만 원본 테두리로 복구한다."""
    height, width = gray.shape
    if min(height, width) < 480:
        return mask
    scale = 480 / min(height, width)
    size = (round(width * scale), round(height * scale))
    rims = _coin_rims(cv2.resize(gray, size, interpolation=cv2.INTER_AREA))
    if not rims:
        return mask

    centers = [(x * width / size[0], y * height / size[1], r / scale) for x, y, r in rims]
    weak = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < mask.size * 0.001 or x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        rim_count = sum(
            cv2.pointPolygonTest(contour, (cx, cy), False) >= 0 for cx, cy, _ in centers
        )
        if rim_count >= 2 or 4 * np.pi * area < 0.75 * perimeter**2:
            weak.append(contour)
    if not weak:
        return mask

    weak_pixels = np.zeros_like(mask)
    cv2.drawContours(weak_pixels, weak, -1, 255, cv2.FILLED)
    weak_pixels &= mask
    kept = mask & ~weak_pixels
    coverage = np.zeros_like(mask)
    candidates = []
    margin = max(1, round(5 / scale))
    for x, y, radius in centers:
        cx, cy, r = round(x), round(y), round(radius)
        extent = r + margin
        roi = np.s_[cy - extent : cy + extent + 1, cx - extent : cx + extent + 1]
        if cx - extent < 0 or cy - extent < 0 or cx + extent >= width or cy + extent >= height:
            continue
        disk = np.zeros((2 * extent + 1, 2 * extent + 1), dtype=np.uint8)
        cv2.circle(disk, (extent, extent), r, 255, cv2.FILLED)
        support = np.zeros_like(disk)
        cv2.circle(support, (extent, extent), extent, 255, cv2.FILLED)
        area = cv2.countNonZero(disk)
        # 반사광으로 마스크가 테두리 바깥에만 남아 있어도 복구 후보로 본다.
        if cv2.countNonZero(support & weak_pixels[roi]) < 0.05 * area:
            continue
        if cv2.countNonZero(disk & kept[roi]) > 0.25 * area:
            continue
        candidates.append((roi, disk, support))
        cv2.circle(coverage, (cx, cy), extent, 255, cv2.FILLED)

    replaced = np.zeros_like(mask)
    for contour in weak:
        x, y, w, h = cv2.boundingRect(contour)
        roi = np.s_[y : y + h, x : x + w]
        region = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        region &= mask[roi]
        # 반사광이 만든 외곽 번짐을 감안하되, 겹침이 적은 영역은 보존한다.
        if cv2.countNonZero(region & coverage[roi]) >= 0.42 * cv2.countNonZero(region):
            replaced[roi] |= region

    result = mask & ~replaced
    # 복구한 원과 기존 전경 사이에 틈을 둬 동전이 다시 붙지 않게 한다.
    gap = max(1, round(0.8 / scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    for roi, disk, support in candidates:
        if cv2.countNonZero(replaced[roi] & support):
            result[roi] |= disk & ~cv2.dilate(result[roi], kernel)
    return result


def preprocess_tune(method: str = "adaptive") -> int:
    """조정용 사진을 처리하고 저장한 PNG 마스크 수를 반환한다."""
    source = DATA_DIR / "raw" / "tune"
    output = DATA_DIR / "processed" / "tune"
    images = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"조정용 사진을 찾을 수 없습니다: {source}")

    for path in images:
        raw = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
        if image is None:
            raise OSError(f"사진을 읽을 수 없습니다: {path}")
        mask = preprocess_image(image, method)
        target = output / path.relative_to(source).with_suffix(path.suffix + ".png")
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise OSError(f"마스크를 인코딩할 수 없습니다: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = NamedTemporaryFile(dir=target.parent, delete=False)
        temporary_path = Path(temporary.name)
        try:
            with temporary:
                temporary.write(encoded.tobytes())
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="조정용 사진의 동전 마스크를 생성합니다.")
    parser.add_argument("--method", choices=("adaptive", "otsu"), default="adaptive")
    args = parser.parse_args()
    try:
        count = preprocess_tune(args.method)
    except OSError as error:
        parser.exit(1, f"{error}\n")
    print(f"조정용 마스크 {count}장 저장: {DATA_DIR / 'processed' / 'tune'}")


# 모듈을 직접 실행할 때만 조정용 사진을 일괄 처리한다.
if __name__ == "__main__":
    main()
