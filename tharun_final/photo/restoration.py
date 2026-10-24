"""Unified classical photo restoration engine.

The public API intentionally lives in this one module so the Streamlit app,
tests, and command-line helper all use the same detection and restoration
implementation.  Crack candidates are refined once; compact dust candidates
are unioned afterwards so crack-specific geometry rules cannot erase them.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


__all__ = [
    "DetectionResult",
    "RestorationResult",
    "detect_cracks",
    "detect_dust",
    "detect_damage",
    "restore_from_mask",
    "restore_image",
]


# ── Internal constants ────────────────────────────────────────────────────────
_MAX_COVERAGE: float = 0.20    # warn if mask exceeds this fraction of the image
_TINY_COVERAGE: float = 0.005  # below this use fast Navier-Stokes directly


_MASK_STAGE_TOKENS = (
    "mask",
    "candidate",
    "seed",
    "hysteresis",
    "frangi_binary",
    "frangi_gated",
    "crack",
    "dust",
    "damage",
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Output of ``detect_damage``; pass directly to ``restore_from_mask``."""
    mask_np: np.ndarray        # uint8 {0, 255} at original image resolution
    mask_image: Image.Image    # PIL L-mode mask for display / download
    coverage: float            # fraction of image pixels in the mask
    warning: str | None = None
    saved_steps: dict[str, str] = field(default_factory=dict)
    debug_dir: str | None = None


@dataclass
class RestorationResult:
    """Final output returned to the UI / API caller."""
    image: Image.Image
    mask: Image.Image
    mask_coverage: float
    steps: list[str]
    warning: str | None = None
    restoration_applied: bool = False
    backend_used: str = "none"
    saved_steps: dict[str, str] = field(default_factory=dict)
    debug_dir: str | None = None


# ── Image Conversions & Helpers ───────────────────────────────────────────────

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_luminance(gray: np.ndarray) -> np.ndarray:
    return np.clip(gray, 0, 255).astype(np.uint8, copy=True)


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(((mask > 0).astype(np.uint8) * 255), mode="L")


def _make_red_overlay_bgr(original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = original_bgr.copy()
    red = np.zeros_like(original_bgr)
    red[:, :] = [30, 30, 235]  # BGR bright red
    active = mask > 0
    if np.any(active):
        overlay[active] = cv2.addWeighted(original_bgr[active], 0.35, red[active], 0.65, 0)
    return overlay


def _generate_step_index(saved_steps: dict[str, str], output_dir: Path) -> None:
    lines = [
        "# Process Step Analysis",
        "",
        "Intermediate step images captured during photo restoration.",
        "",
        "| Step Name | Image File |",
        "|---|---|",
    ]
    for name, path_str in sorted(saved_steps.items()):
        fname = Path(path_str).name
        lines.append(f"| `{name}` | [`{fname}`]({fname}) |")
    lines.append("")
    lines.extend(
        [
            "## Auditing the masks",
            "",
            "- `MASK_METRICS.md` / `MASK_METRICS.json` report shape, pixel count, coverage, and connected components for every detection stage.",
            "- Mask stages also have a corresponding `_overlay.png` showing the stage in red over the original image.",
            "- `RESTORATION_METRICS.md` / `RESTORATION_METRICS.json` report the inpainting and final-composite stages.",
            "",
        ]
    )
    try:
        (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _is_mask_stage(name: str, data: np.ndarray) -> bool:
    """Return whether an intermediate array should also get a red overlay."""
    if data.ndim != 2:
        return False
    if not (data.dtype == bool or np.issubdtype(data.dtype, np.integer)):
        return False
    return any(token in name.lower() for token in _MASK_STAGE_TOKENS)


def _step_metrics(name: str, data: np.ndarray | Image.Image) -> dict[str, Any]:
    """Describe an intermediate image so mask size can be audited numerically."""
    if isinstance(data, Image.Image):
        array = np.asarray(data)
    else:
        array = np.asarray(data)

    metrics: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.nanmin(array)) if array.size else None,
        "max": float(np.nanmax(array)) if array.size else None,
    }
    if array.ndim == 2:
        positive = array > 0
        metrics["nonzero_pixels"] = int(np.count_nonzero(positive))
        metrics["coverage"] = float(np.mean(positive)) if array.size else 0.0
        if _is_mask_stage(name, array):
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                positive.astype(np.uint8), connectivity=8
            )
            areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int32)
            metrics["components"] = int(max(0, count - 1))
            metrics["largest_component_pixels"] = int(areas.max()) if areas.size else 0
    return metrics


def _write_step_metrics(
    steps_dict: Mapping[str, np.ndarray | Image.Image],
    output_dir: Path,
    filename_stem: str = "MASK_METRICS",
) -> None:
    """Write a machine-readable and human-readable audit of every saved step."""
    metrics: dict[str, dict[str, Any]] = {
        name: _step_metrics(name, data) for name, data in steps_dict.items()
    }
    try:
        (output_dir / f"{filename_stem}.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        lines = [
            "# Mask and intermediate-image metrics",
            "",
            "All arrays are saved at the same source-image resolution. `coverage` "
            "is the fraction of pixels with a value greater than zero.",
            "",
            "| Step | Shape | Dtype | Non-zero | Coverage | Components | Largest component |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for name, item in metrics.items():
            shape_value = item.get("shape", [])
            shape_text = (
                "x".join(str(value) for value in shape_value)
                if isinstance(shape_value, list)
                else "?"
            )
            coverage_value = item.get("coverage", 0.0)
            coverage_number = float(coverage_value) if isinstance(coverage_value, (int, float)) else 0.0
            lines.append(
                "| `{name}` | `{shape}` | `{dtype}` | {nonzero} | {coverage:.2%} | "
                "{components} | {largest} |".format(
                    name=name,
                    shape=shape_text,
                    dtype=item["dtype"],
                    nonzero=item.get("nonzero_pixels", "—"),
                    coverage=coverage_number,
                    components=item.get("components", "—"),
                    largest=item.get("largest_component_pixels", "—"),
                )
            )
        (output_dir / f"{filename_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _save_step_image(name: str, data: np.ndarray | Image.Image, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.png"
    if isinstance(data, Image.Image):
        data.save(out_path)
        return str(out_path)
    arr = np.asarray(data)
    if arr.dtype == bool:
        cv2.imwrite(str(out_path), (arr.astype(np.uint8) * 255))
    elif arr.ndim == 2 and arr.dtype in (np.float32, np.float64):
        finite = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        minimum, maximum = float(finite.min()), float(finite.max())
        if minimum < 0.0 < maximum:
            scale = max(abs(minimum), abs(maximum), 1e-6)
            vis = np.clip(128.0 + finite * (127.0 / scale), 0, 255).astype(np.uint8)
        else:
            scale = max(abs(minimum), abs(maximum), 1e-6)
            vis = np.clip((finite / scale) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), vis)
    else:
        cv2.imwrite(str(out_path), arr)
    return str(out_path)


def _save_step_overlay(
    name: str,
    data: np.ndarray,
    original_bgr: np.ndarray,
    output_dir: Path,
) -> str | None:
    """Save a binary intermediate as a red overlay on the original image."""
    if not _is_mask_stage(name, data):
        return None
    binary = _as_binary(data)
    return _save_step_image(
        f"{name}_overlay",
        _make_red_overlay_bgr(original_bgr, binary),
        output_dir,
    )


# ── Denoising & Enhancement ───────────────────────────────────────────────────

def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Localized grain reduction / denoising."""
    if strength <= 0:
        return image
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


# ── Inpainting ────────────────────────────────────────────────────────────────

def inpaint(image: np.ndarray, mask: np.ndarray, method: str = "ns", radius: float = 3.0) -> np.ndarray:
    """Fast classical inpainting via OpenCV (Telea or Navier-Stokes)."""
    if image.ndim not in (2, 3) or mask.ndim != 2:
        raise ValueError("image must be 2D/3D and mask must be 2D")
    if image.shape[:2] != mask.shape:
        raise ValueError("image and mask dimensions must match")
    if method not in {"telea", "ns"}:
        raise ValueError("method must be 'telea' or 'ns'")
    binary = ((mask > 0).astype(np.uint8) * 255)
    if not np.any(binary):
        return image.copy()
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(image, binary, float(np.clip(radius, 1.0, 15.0)), flag)


def inpaint_high_quality(image: np.ndarray, mask: np.ndarray, max_dim: int = 1024) -> tuple[np.ndarray, str]:
    """High quality inpainting. Uses biharmonic PDE when skimage is available,

    otherwise falls back to OpenCV Navier-Stokes and honestly returns backend name.
    """
    binary_mask = (mask > 0)
    if not np.any(binary_mask):
        return image.copy(), "none"

    try:
        from skimage.restoration import inpaint_biharmonic
        h, w = image.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            new_w = max(32, int(round(w * scale)))
            new_h = max(32, int(round(h * scale)))
            work_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            work_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            work_img = image
            work_mask = binary_mask

        img_float = work_img.astype(np.float64) / 255.0
        if image.ndim == 3:
            result_float = inpaint_biharmonic(img_float, work_mask, channel_axis=-1)
        else:
            result_float = inpaint_biharmonic(img_float, work_mask)

        result = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
        if scale < 1.0:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
        return result, "biharmonic"

    except ImportError:
        # Fall back cleanly to Navier-Stokes
        return inpaint(image, mask, method="ns", radius=4.0), "Navier–Stokes"


# ── Crack Detection Math & Filtering Helpers ──────────────────────────────────

def _line_kernel(length: int, angle: int) -> np.ndarray:
    size = max(3, int(length) | 1)
    kernel = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    radians = np.deg2rad(angle)
    dx = int(np.cos(radians) * centre)
    dy = int(np.sin(radians) * centre)
    cv2.line(kernel, (centre - dx, centre - dy), (centre + dx, centre + dy), 1, 1)
    return kernel


def _robust_threshold(values: np.ndarray, sensitivity: float) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 255.0
    level = float(np.clip((sensitivity - 0.25) / 1.75, 0.0, 1.0))
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    quantile = float(np.quantile(finite, 0.985 - 0.055 * level))
    return min(float(np.max(finite)), max(7.0 / sensitivity, quantile, median + 3.0 * mad / sensitivity))


def _contrast_threshold(values: np.ndarray, sensitivity: float) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 255.0
    level = float(np.clip((sensitivity - 0.25) / 1.75, 0.0, 1.0))
    quantile = float(np.quantile(positive, 0.985 - 0.055 * level))
    return max(8.0 / sensitivity, quantile * 0.78)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _noise_level(gray: np.ndarray) -> float:
    values = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(values, (0, 0), sigmaX=2.0)
    residual = np.abs(values - smooth)
    median = float(np.median(residual))
    return max(0.5, 1.4826 * median)


def _close_fragments(mask: np.ndarray, sensitivity: float) -> np.ndarray:
    if not np.any(mask):
        return mask
    level = float(np.clip((sensitivity - 0.25) / 1.75, 0.0, 1.0))
    max_gap = 1 + int(round(level * 2.0))
    result = mask.copy()
    for angle in range(0, 180, 15):
        for length in (3, 3 + 2 * max_gap):
            kernel = _line_kernel(length, angle)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            result = np.maximum(result, closed)
    return result


def _as_binary(mask: np.ndarray) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255).astype(np.uint8)


def _compute_frangi_native(gray: np.ndarray, sensitivity: float, detect_dark: bool) -> tuple[np.ndarray, np.ndarray]:
    float_img = gray.astype(np.float32)
    max_vesselness = np.zeros_like(float_img)
    level = float(np.clip((sensitivity - 0.25) / 1.75, 0.0, 1.0))
    beta = 0.5
    c = 15.0

    for s in range(1, 7):
        ksize = int(2 * round(3 * s) + 1)
        smoothed = cv2.GaussianBlur(float_img, (ksize, ksize), float(s))
        s2 = float(s ** 2)
        Dxx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3) * s2
        Dyy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3) * s2
        Dxy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3) * s2

        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy * Dxy
        disc = np.maximum(0.0, trace * trace - 4.0 * det)
        sqrt_disc = np.sqrt(disc)

        l1 = 0.5 * (trace - sqrt_disc)
        l2 = 0.5 * (trace + sqrt_disc)

        swap = np.abs(l1) > np.abs(l2)
        e1 = np.where(swap, l2, l1)
        e2 = np.where(swap, l1, l2)

        condition = (e2 > 0) if detect_dark else (e2 < 0)
        Rb = np.abs(e1) / (np.abs(e2) + 1e-8)
        S2 = e1 * e1 + e2 * e2

        vesselness = np.exp(-(Rb ** 2) / (2.0 * (beta ** 2))) * (1.0 - np.exp(-S2 / (2.0 * (c ** 2))))
        vesselness = np.where(condition, vesselness, 0.0)
        max_vesselness = np.maximum(max_vesselness, vesselness)

    pos = max_vesselness[max_vesselness > 0]
    if pos.size == 0:
        return np.zeros(gray.shape[:2], dtype=np.uint8), max_vesselness

    q = float(np.clip(0.97 - 0.04 * level, 0.90, 0.98))
    thresh = float(np.quantile(pos, q))
    binary = ((max_vesselness >= thresh).astype(np.uint8) * 255)
    return binary, max_vesselness


def _compute_frangi(source: np.ndarray, sensitivity: float, detect_dark: bool) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.filters import frangi
        gray_float = source.astype(np.float64) / 255.0
        level = float(np.clip((sensitivity - 0.25) / 1.75, 0.0, 1.0))
        vesselness = frangi(gray_float, sigmas=range(1, 7), black_ridges=False, mode="reflect")
        if detect_dark:
            dark_v = frangi(gray_float, sigmas=range(1, 7), black_ridges=True, mode="reflect")
            vesselness = np.maximum(vesselness, dark_v)
        pos = vesselness[vesselness > 0]
        if pos.size == 0:
            return np.zeros(source.shape[:2], dtype=np.uint8), vesselness.astype(np.float32)
        q = float(np.clip(0.97 - 0.04 * level, 0.90, 0.98))
        thresh = float(np.quantile(pos, q))
        binary = ((vesselness >= thresh).astype(np.uint8) * 255)
        return binary, vesselness.astype(np.float32)
    except ImportError:
        return _compute_frangi_native(source, sensitivity, detect_dark)


def _elongation(component: np.ndarray) -> float:
    points = np.column_stack(np.where(component > 0))
    if len(points) < 3:
        return 1.0
    covariance = np.cov(points.astype(np.float32), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float((eigenvalues[-1] + 1e-3) / (eigenvalues[0] + 1e-3))


def refine_mask(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
    strength: float = 1.0,
    gray: np.ndarray | None = None,
    *,
    sensitivity: float | None = None,
    expansion: float | None = None,
    preserve_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Filter broad non-crack blobs and enforce crack-like geometry.

    ``strength`` and ``expansion`` remain accepted for compatibility with the
    old split-module API, but mask expansion is intentionally disabled.  A
    supplied ``preserve_mask`` is used for high-confidence ridge pixels that
    would otherwise be discarded only because their connected component is
    small. This keeps fine Frangi crack detail without sending dust through
    crack-specific filtering.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("refine_mask expects a non-empty two-dimensional mask")
    if tuple(image_shape[:2]) != mask.shape:
        raise ValueError("image_shape must match the mask dimensions")
    if gray is not None:
        gray = np.asarray(gray)
        if gray.ndim != 2 or gray.shape != mask.shape:
            raise ValueError("gray must match the mask dimensions")
        gray = np.clip(np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
    if preserve_mask is not None:
        preserve_mask = np.asarray(preserve_mask)
        if preserve_mask.ndim != 2 or preserve_mask.shape != mask.shape:
            raise ValueError("preserve_mask must match the mask dimensions")

    sensitivity_value = float(1.0 if sensitivity is None else sensitivity)
    sensitivity_value = float(np.clip(sensitivity_value, 0.25, 2.0))

    binary = ((np.isfinite(mask) & (mask > 0)).astype(np.uint8) * 255)
    if not np.any(binary):
        return binary

    # Close small response gaps
    joined = binary.copy()
    gap_length = 3 if sensitivity_value < 1.5 else 5
    for kernel in (
        np.ones((1, gap_length), np.uint8),
        np.ones((gap_length, 1), np.uint8),
        np.eye(gap_length, dtype=np.uint8),
        np.fliplr(np.eye(gap_length, dtype=np.uint8)),
    ):
        joined = cv2.bitwise_or(joined, cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))
    binary = joined

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = mask.shape
    min_area = max(3, int(round(8.0 / sensitivity_value)))
    kept = np.zeros_like(binary)

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        box_w, box_h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        box_area = max(1, box_w * box_h)
        fill = area / box_area

        margin = 4
        left, top = max(0, x - margin), max(0, y - margin)
        right, bottom = min(width, x + box_w + margin), min(height, y + box_h + margin)
        component = (labels[top:bottom, left:right] == index).astype(np.uint8)
        max_dim = max(box_w, box_h)

        if area < min_area or max_dim < 6:
            continue

        distances = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        thickness = 2.0 * float(np.quantile(distances[component > 0], 0.85))
        if thickness > 20.0:
            continue

        elongation = _elongation(component)
        crack_like = (
            (elongation >= 3.0 and max_dim >= max(7.0, thickness * 2.5))
            or (fill <= 0.40 and max_dim >= max(9.0, thickness * 3.0))
        )
        if not crack_like:
            continue

        if gray is not None:
            surround = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            ring = surround & (binary[top:bottom, left:right] == 0)
            if not np.any(ring):
                continue
            pixels = gray[top:bottom, left:right]
            inner = pixels[component > 0]
            outer = pixels[ring]
            contrast = max(
                float(np.quantile(inner, 0.60)) - float(np.quantile(outer, 0.75)),
                float(np.quantile(outer, 0.25)) - float(np.quantile(inner, 0.40)),
            )
            if contrast < 5.0 / sensitivity_value and thickness < 2.0:
                continue

        region = kept[top:bottom, left:right]
        region[component > 0] = 255
        kept[top:bottom, left:right] = region

    if preserve_mask is not None:
        # Frangi is already thresholded and flank-gated before it reaches this
        # function. Preserve those high-confidence ridge pixels even when a
        # fine crack fragment is too small for the component geometry rules.
        kept = cv2.bitwise_or(kept, _as_binary(preserve_mask))

    return ((kept > 0).astype(np.uint8) * 255).astype(np.uint8)


# ── Crack & Dust Detectors ───────────────────────────────────────────────────

def detect_dust(
    gray: np.ndarray,
    sensitivity: float = 1.0,
    steps_dict: MutableMapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Detect compact dust/spot blemishes without turning texture into dust.

    Dust is intentionally conservative: the response threshold is high, the
    connected component must remain small and compact, and elongated or
    low-fill components are rejected. ``steps_dict`` exposes the response,
    thresholded candidates, and final mask for process inspection.
    """
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("detect_dust expects a non-empty grayscale image")
    sensitivity = float(np.clip(sensitivity, 0.25, 2.0))
    source = np.clip(gray, 0, 255).astype(np.uint8)
    response = np.zeros_like(source, dtype=np.float32)

    for size in (3, 5, 7):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        bright = cv2.morphologyEx(source, cv2.MORPH_TOPHAT, kernel)
        dark = cv2.morphologyEx(source, cv2.MORPH_BLACKHAT, kernel)
        response = np.maximum(response, np.maximum(bright, dark).astype(np.float32))

    positive = response[response > 0]
    if positive.size == 0:
        if steps_dict is not None:
            empty = np.zeros_like(source, dtype=np.float32)
            empty_mask = np.zeros_like(source, dtype=np.uint8)
            steps_dict["08a_dust_response_raw"] = empty
            steps_dict["08b_dust_response_thresholded"] = empty
            steps_dict["08c_dust_candidates"] = empty_mask
            steps_dict["08_dust_mask"] = empty_mask
        return np.zeros_like(source, dtype=np.uint8)

    level = float(np.clip((sensitivity - 0.25) / 1.75, 0.0, 1.0))
    response_quantile = float(np.clip(0.985 - 0.01 * level, 0.97, 0.99))
    threshold = max(
        24.0 / sensitivity,
        float(np.quantile(positive, response_quantile)),
    )
    candidates = (response >= threshold).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    result = np.zeros_like(candidates)
    image_height, image_width = source.shape
    image_area = image_height * image_width
    max_area = max(64, int(image_area * 0.0008))
    max_dimension = max(8, min(14, int(round(min(image_height, image_width) * 0.05))))

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if not (2 <= area <= max_area):
            continue
        if max(width, height) > max_dimension or min(width, height) < 1:
            continue
        fill = area / max(1, width * height)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        if fill < 0.22 or aspect_ratio > 3.5:
            continue

        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        compactness = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        if compactness < 0.12:
            continue
        result[labels == index] = 255

    if steps_dict is not None:
        steps_dict["08a_dust_response_raw"] = response
        steps_dict["08b_dust_response_thresholded"] = np.where(
            response >= threshold,
            response,
            0.0,
        ).astype(np.float32)
        steps_dict["08c_dust_candidates"] = candidates
        steps_dict["08_dust_mask"] = result

    return result.astype(np.uint8)


def detect_cracks(
    gray: np.ndarray,
    strength: float = 1.0,
    sensitivity: float | None = None,
    detect_dark: bool = False,
    *,
    include_dark: bool | None = None,
    dark: bool | None = None,
    expansion: float = 0.0,
    steps_dict: MutableMapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Detect line-like bright cracks and optional dark cracks."""
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("detect_cracks expects a non-empty grayscale image")
    if include_dark is not None:
        detect_dark = bool(include_dark)
    if dark is not None:
        detect_dark = bool(dark)

    selected_sensitivity = strength if sensitivity is None else sensitivity
    sensitivity_value = float(np.clip(float(selected_sensitivity), 0.25, 2.0))

    clean = np.clip(np.nan_to_num(np.asarray(gray), nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
    source = clean.astype(np.float32)
    smooth = cv2.GaussianBlur(source, (3, 3), 0)
    background = cv2.GaussianBlur(source, (0, 0), sigmaX=8.0)
    noise = _noise_level(clean)
    local_contrast = source - background
    contrast_threshold = max(_contrast_threshold(np.abs(local_contrast), sensitivity_value), noise * 1.5 / sensitivity_value)
    bright_pixels = local_contrast >= contrast_threshold
    dark_pixels = local_contrast <= -contrast_threshold

    bright_response = np.zeros_like(smooth, dtype=np.float32)
    dark_response = np.zeros_like(smooth, dtype=np.float32)

    # Orientation sweep: build response maps
    shifts_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for radius in (2, 3, 5, 8, 12):
        for angle in range(0, 180, 15):
            radians = np.deg2rad(angle)
            dx = int(round(-np.sin(radians) * radius))
            dy = int(round(np.cos(radians) * radius))
            side_a = _shift(smooth, dx, dy)
            side_b = _shift(smooth, -dx, -dy)
            shifts_cache[(radius, angle)] = (side_a, side_b)
            bright_response = np.maximum(bright_response, np.minimum(smooth - side_a, smooth - side_b))
            if detect_dark:
                dark_response = np.maximum(dark_response, np.minimum(side_a - smooth, side_b - smooth))

    bright_threshold = max(_robust_threshold(bright_response, sensitivity_value), noise * 2.5 / sensitivity_value)
    dark_threshold = max(_robust_threshold(dark_response, sensitivity_value), noise * 2.5 / sensitivity_value)
    if bright_response.max() > 0:
        bright_threshold = min(bright_threshold, float(bright_response.max()) * 0.75)
    if dark_response.max() > 0:
        dark_threshold = min(dark_threshold, float(dark_response.max()) * 0.75)

    candidates = np.zeros_like(clean, dtype=np.uint8)
    seeds = np.zeros_like(clean, dtype=np.uint8)

    # Reuse cached shifted arrays for longitudinal continuity
    for radius in (2, 3, 5, 8, 12):
        for angle in range(0, 180, 15):
            side_a, side_b = shifts_cache[(radius, angle)]
            responses = [(np.minimum(smooth - side_a, smooth - side_b), bright_pixels, bright_threshold)]
            if detect_dark:
                responses.append((np.minimum(side_a - smooth, side_b - smooth), dark_pixels, dark_threshold))
            kernel = _line_kernel(max(7, radius * 4 + 1), angle).astype(np.float32)
            kernel /= kernel.sum()
            for response, local_pixels, response_thresh in responses:
                evidence = (response >= response_thresh) & local_pixels
                along = cv2.filter2D(evidence.astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
                candidates |= evidence.astype(np.uint8)
                seeds |= (evidence & (along >= 0.60)).astype(np.uint8)

    raw_candidates = candidates.copy()

    # Hysteresis: keep components containing at least one seed pixel
    count, labels = cv2.connectedComponents(candidates, connectivity=8)
    supported = np.zeros(count, dtype=np.uint8)
    supported[np.unique(labels[seeds > 0])] = 1
    supported[0] = 0
    candidates = supported[labels]
    hysteresis_candidates = candidates.copy()

    # Close small response gaps
    symmetry_binary = _as_binary(_close_fragments(candidates, sensitivity_value))

    # Frangi vesselness filter
    frangi_binary, frangi_vesselness = _compute_frangi(clean, sensitivity_value, detect_dark)
    flank_support = (bright_response >= max(1.0, noise * 1.5 / sensitivity_value))
    if detect_dark:
        flank_support |= (dark_response >= max(1.0, noise * 1.5 / sensitivity_value))
    frangi_gated = np.where(flank_support, frangi_binary, 0).astype(np.uint8)

    combined = cv2.bitwise_or(symmetry_binary, frangi_gated)

    # Single geometric refinement pass on crack candidates
    refined = refine_mask(
        combined,
        clean.shape,
        gray=clean,
        sensitivity=sensitivity_value,
        expansion=expansion,
        preserve_mask=frangi_gated,
    )

    if steps_dict is not None:
        steps_dict["03_smooth"] = smooth
        steps_dict["04_local_contrast"] = local_contrast
        steps_dict["05a_symmetry_bright_response"] = bright_response
        steps_dict["05b_symmetry_candidates"] = _as_binary(raw_candidates)
        steps_dict["05c_symmetry_seeds"] = _as_binary(seeds)
        steps_dict["05d_symmetry_hysteresis"] = _as_binary(hysteresis_candidates)
        steps_dict["05e_symmetry_closed"] = symmetry_binary
        steps_dict["06a_frangi_vesselness"] = frangi_vesselness
        steps_dict["06b_frangi_binary"] = frangi_binary
        steps_dict["06_frangi_ridges"] = frangi_gated
        steps_dict["07a_combined_crack_candidates"] = combined
        steps_dict["07_crack_mask"] = refined

    return refined


# ── Damage Detection Orchestration ───────────────────────────────────────────

def detect_damage(
    image: Image.Image,
    sensitivity: float = 1.0,
    bright_cracks: bool = True,
    dark_cracks: bool = False,
    detect_dust_marks: bool = False,
    debug_dir: str | Path | None = None,
    expansion: float = 0.0,
) -> DetectionResult:
    """Stage 1: Detect cracks and dust with one clean mask-combination pass.

    ``expansion`` is retained as a no-op compatibility argument.  The unified
    pipeline never dilates the final mask, because a broad seam allowance can
    erase genuine compact dust and undamaged surrounding detail.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    sensitivity = float(np.clip(sensitivity, 0.4, 1.8))
    original_bgr = pil_to_bgr(image)
    gray = normalize_luminance(luminance(original_bgr))

    steps_dict: dict[str, np.ndarray] | None = {} if debug_dir is not None else None
    if steps_dict is not None:
        steps_dict["01_original_input"] = original_bgr
        steps_dict["02_grayscale_luminance"] = gray

    # 1. Crack detection (geometric filtering is done once inside detect_cracks)
    crack_mask = np.zeros_like(gray, dtype=np.uint8)
    if bright_cracks or dark_cracks:
        crack_mask = detect_cracks(
            gray,
            sensitivity=sensitivity,
            detect_dark=dark_cracks,
            expansion=0.0,
            steps_dict=steps_dict,
        )
    if crack_mask.shape != gray.shape:
        raise RuntimeError(
            f"crack mask shape {crack_mask.shape} does not match image shape {gray.shape}"
        )

    # 2. Dust detection (kept clean, never destroyed by crack elongation filtering)
    dust_mask = np.zeros_like(gray, dtype=np.uint8)
    if detect_dust_marks:
        raw_dust_mask = detect_dust(gray, sensitivity=sensitivity, steps_dict=steps_dict)
        dust_mask = cv2.bitwise_and(raw_dust_mask, cv2.bitwise_not(crack_mask))
        if steps_dict is not None:
            steps_dict["08_dust_mask_raw"] = raw_dust_mask
            steps_dict["08_dust_mask"] = dust_mask
    if dust_mask.shape != gray.shape:
        raise RuntimeError(
            f"dust mask shape {dust_mask.shape} does not match image shape {gray.shape}"
        )

    # 3. Direct union — NO redundant second refine_mask!
    final_mask = cv2.bitwise_or(crack_mask, dust_mask)

    coverage = float(np.count_nonzero(final_mask)) / final_mask.size
    mask_img = _mask_image(final_mask)

    saved_steps: dict[str, str] = {}
    if debug_dir is not None and steps_dict is not None:
        out_dir = Path(debug_dir)
        steps_dict["09_damage_mask"] = final_mask
        steps_dict["10_damage_overlay_red"] = _make_red_overlay_bgr(original_bgr, final_mask)
        for s_name, s_data in steps_dict.items():
            saved_steps[s_name] = _save_step_image(s_name, s_data, out_dir)
            if isinstance(s_data, np.ndarray):
                overlay_path = _save_step_overlay(s_name, s_data, original_bgr, out_dir)
                if overlay_path is not None:
                    saved_steps[f"{s_name}_overlay"] = overlay_path
        _write_step_metrics(steps_dict, out_dir)
        _generate_step_index(saved_steps, out_dir)

    warning: str | None = None
    if coverage == 0:
        warning = "No damage detected. Try raising the sensitivity slider."
    elif coverage > _MAX_COVERAGE:
        warning = (
            f"Mask covers {coverage:.1%} of the image — unusually large. "
            "Lower the sensitivity slider or uncheck damage types to reduce false positives."
        )

    return DetectionResult(
        mask_np=final_mask,
        mask_image=mask_img,
        coverage=coverage,
        warning=warning,
        saved_steps=saved_steps,
        debug_dir=str(debug_dir) if debug_dir else None,
    )


# ── Restoration & Inpainting Orchestration ───────────────────────────────────

def restore_from_mask(
    image: Image.Image,
    detection: DetectionResult,
    denoise_strength: int = 0,
    debug_dir: str | Path | None = None,
) -> RestorationResult:
    """Stage 2: Inpaint detected mask regions while keeping undamaged pixels exact."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    active_debug_dir = debug_dir or detection.debug_dir
    saved_steps: dict[str, str] = dict(detection.saved_steps)

    if detection.coverage == 0:
        return RestorationResult(
            image=image.copy(),
            mask=detection.mask_image,
            mask_coverage=0.0,
            steps=["no damage detected — original returned"],
            warning="No damage was detected; the image is unchanged.",
            saved_steps=saved_steps,
            debug_dir=str(active_debug_dir) if active_debug_dir else None,
        )

    original_bgr = pil_to_bgr(image)
    final_mask = detection.mask_np
    steps: list[str] = []

    if detection.coverage < _TINY_COVERAGE:
        restored = inpaint(original_bgr, final_mask, method="ns", radius=3.0)
        backend_used = "Navier–Stokes"
        steps.append("Navier–Stokes inpainting (fast path)")
    else:
        restored, backend_used = inpaint_high_quality(original_bgr, final_mask)
        steps.append(f"{backend_used} inpainting")

    # Local composite: strictly preserve 100% of pixels outside the mask
    output = original_bgr.copy()
    denoised_img: np.ndarray | None = None
    if denoise_strength > 0:
        denoised_img = denoise(restored, int(np.clip(denoise_strength, 1, 10)))
        output[final_mask > 0] = denoised_img[final_mask > 0]
        steps.append(f"local grain reduction (strength {denoise_strength})")
    else:
        output[final_mask > 0] = restored[final_mask > 0]

    steps.append("localized compositing — outside-mask pixels preserved exactly")

    if active_debug_dir is not None:
        out_dir = Path(active_debug_dir)
        saved_steps["11_inpainting_output_raw"] = _save_step_image("11_inpainting_output_raw", restored, out_dir)
        if denoised_img is not None:
            saved_steps["11a_grain_reduction_denoised"] = _save_step_image(
                "11a_grain_reduction_denoised", denoised_img, out_dir
            )
        saved_steps["12_final_restored_composite"] = _save_step_image("12_final_restored_composite", output, out_dir)
        restoration_steps: dict[str, np.ndarray | Image.Image] = {
            "11_inpainting_output_raw": restored,
            "12_final_restored_composite": output,
        }
        if denoised_img is not None:
            restoration_steps["11a_grain_reduction_denoised"] = denoised_img
        _write_step_metrics(restoration_steps, out_dir, filename_stem="RESTORATION_METRICS")
        _generate_step_index(saved_steps, out_dir)

    return RestorationResult(
        image=bgr_to_pil(output),
        mask=detection.mask_image,
        mask_coverage=detection.coverage,
        steps=steps,
        warning=None,
        restoration_applied=True,
        backend_used=backend_used,
        saved_steps=saved_steps,
        debug_dir=str(active_debug_dir) if active_debug_dir else None,
    )


def restore_image(
    image: Image.Image,
    sensitivity: float = 1.0,
    denoise_strength: int = 0,
    bright_cracks: bool = True,
    dark_cracks: bool = False,
    detect_dust_marks: bool = False,
    debug_dir: str | Path | None = None,
    **_legacy_kwargs,
) -> RestorationResult:
    """Convenience function: detect + restore in one call."""
    detection = detect_damage(
        image,
        sensitivity=sensitivity,
        bright_cracks=bright_cracks,
        dark_cracks=dark_cracks,
        detect_dust_marks=detect_dust_marks,
        debug_dir=debug_dir,
    )
    return restore_from_mask(
        image,
        detection,
        denoise_strength=denoise_strength,
        debug_dir=debug_dir,
    )
