"""Compare denoising algorithms and visualize removed details."""

from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_processor import SUPPORTED_METHODS, denoise_image


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "test_images" / "old_photo.jpg"
OUTPUT_DIR = ROOT / "outputs"
STRENGTH = 58


def create_noise_visualization(original, restored):
    """
    Create a high-contrast residual visualization.
    Dark background with bright details representing
    differences between the input and restored image.
    """
    difference = cv2.absdiff(original, restored)

    # Use the strongest channel difference at each pixel.
    residual = np.max(difference, axis=2).astype(np.float32)

    # Stretch contrast so subtle differences are visible.
    upper = np.percentile(residual, 99.5)

    if upper > 0:
        residual = np.clip(residual / upper, 0, 1) * 255
    else:
        residual = np.zeros_like(residual)

    return residual.astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(
        str(INPUT_PATH), cv2.IMREAD_COLOR
    )

    if original is None:
        raise FileNotFoundError(
            f"Cannot load {INPUT_PATH}. Place old_photo.jpg "
            "inside the test_images folder."
        )

    results = {}
    timings = {}

    # Run all three algorithms on the same original image.
    for method in SUPPORTED_METHODS:
        start = time.perf_counter()

        restored = denoise_image(
            original,
            method=method,
            strength=STRENGTH
        )

        timings[method] = time.perf_counter() - start
        results[method] = restored

        cv2.imwrite(
            str(OUTPUT_DIR / f"{method}.png"),
            restored
        )

    # Create one residual image per algorithm.
    residuals = {}

    for method in SUPPORTED_METHODS:
        residuals[method] = create_noise_visualization(
            original, results[method]
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{method}_removed_noise.png"),
            residuals[method]
        )

    # Layout: original occupies the first column.
    # Each other column contains restored image + residual.
    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(
        2, 4,
        width_ratios=[1.1, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.12,
        hspace=0.20
    )

    # Original image spans both rows in column 1.
    ax_original = fig.add_subplot(grid[:, 0])
    ax_original.imshow(
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    )
    ax_original.set_title("Original Image", fontsize=13)
    ax_original.axis("off")

    method_titles = {
        "median": "Method 1: Median Filter",
        "gaussian": "Method 2: Gaussian Filter",
        "nlm": "Method 3: Non-Local Means"
    }

    for column, method in enumerate(SUPPORTED_METHODS, start=1):
        # Restored photograph: top row.
        ax_restored = fig.add_subplot(grid[0, column])
        ax_restored.imshow(
            cv2.cvtColor(
                results[method], cv2.COLOR_BGR2RGB
            )
        )
        ax_restored.set_title(
            method_titles[method], fontsize=11
        )
        ax_restored.axis("off")

        # Removed-detail visualization: bottom row.
        ax_noise = fig.add_subplot(grid[1, column])
        ax_noise.imshow(
            residuals[method],
            cmap="gray",
            vmin=0,
            vmax=255
        )
        ax_noise.set_title(
            "Removed Noise / Residual",
            fontsize=10
        )
        ax_noise.axis("off")

    fig.suptitle(
        f"Image Denoising Comparison — Strength {STRENGTH}",
        fontsize=15
    )

    fig.savefig(
        OUTPUT_DIR / "comparison.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.show()

    print("\nProcessing-time comparison")
    print("-" * 36)

    for method in SUPPORTED_METHODS:
        print(f"{method:10s}: {timings[method]:.4f} seconds")

    print(f"\nAll results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()