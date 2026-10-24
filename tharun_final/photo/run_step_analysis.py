#!/usr/bin/env python3
"""Run photo restoration and save all intermediate process images to a folder.

Usage:
    python run_step_analysis.py [image_path] [--output-dir FOLDER] [--sensitivity FLOAT] [--denoise INT]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

from restoration import restore_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Save all intermediate photo restoration step images.")
    parser.add_argument("image", nargs="?", default="images (1).jpeg", help="path to photo to restore (default: images (1).jpeg)")
    parser.add_argument("--output-dir", "-o", default="FINAL_IMAGES", help="directory to save step images (default: FINAL_IMAGES)")
    parser.add_argument("--sensitivity", "-s", type=float, default=1.0, help="mask sensitivity (0.4 to 1.8)")
    parser.add_argument("--denoise", "-d", type=int, default=3, help="grain reduction strength (0 to 10)")
    parser.add_argument("--dark-cracks", action="store_true", help="enable dark crack detection")
    parser.add_argument("--dust", action="store_true", help="enable dust and small spot detection")
    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.is_file():
        raise SystemExit(f"Error: input image not found at {input_path}")

    print(f"Loading {input_path}...")
    image = Image.open(input_path).convert("RGB")

    out_dir = Path(args.output_dir)
    print(f"Running detection and restoration; saving step images to '{out_dir}/'...")

    result = restore_image(
        image,
        sensitivity=args.sensitivity,
        denoise_strength=args.denoise,
        bright_cracks=True,
        dark_cracks=args.dark_cracks,
        detect_dust_marks=args.dust,
        debug_dir=out_dir,
    )

    # Easy-to-find aliases alongside the numbered diagnostic stages.
    image.save(out_dir / "original_input.png")
    result.mask.save(out_dir / "final_damage_mask.png")
    result.image.save(out_dir / "restored_output.png")

    print(f"Finished! Damage mask coverage: {result.mask_coverage:.2%}")
    print(f"Restoration backend used: {result.backend_used}")
    print(f"Saved {len(result.saved_steps)} step images to {out_dir}/:")
    for step_name, file_path in sorted(result.saved_steps.items()):
        print(f"  • {step_name} -> {file_path}")

    print(f"\nSee '{out_dir}/INDEX.md' for full step descriptions.")


if __name__ == "__main__":
    main()
