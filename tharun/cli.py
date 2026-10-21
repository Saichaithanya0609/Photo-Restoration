from __future__ import annotations

import argparse

from PIL import Image

from .pipeline import restore_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore an old photograph.")
    parser.add_argument("input", help="input image path")
    parser.add_argument("output", help="output image path")
    parser.add_argument("--strength", type=float, default=0.9, help="classical inpainting strength")
    parser.add_argument("--sensitivity", type=float, default=0.9)
    parser.add_argument("--method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--backend", choices=["auto", "telea", "ns", "lama"], default="auto")
    parser.add_argument("--model", default=None, help="optional local TorchScript LaMa model")
    parser.add_argument("--dark-cracks", action="store_true")
    parser.add_argument("--dust", action="store_true")
    parser.add_argument("--mask-expansion", type=float, default=1.5)
    parser.add_argument("--max-coverage", type=float, default=0.08)
    parser.add_argument("--denoise", type=int, default=0)
    parser.add_argument("--fading", action="store_true")
    parser.add_argument("--color-balance", action="store_true")
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--steps-dir", default=None, help="directory to save intermediate processing step images")
    args = parser.parse_args()
    result = restore_image(
        Image.open(args.input),
        strength=args.strength,
        method=args.method,
        detection_sensitivity=args.sensitivity,
        dark_cracks=args.dark_cracks,
        detect_dust_marks=args.dust,
        mask_expansion=args.mask_expansion,
        max_coverage=args.max_coverage,
        denoise_strength=args.denoise,
        correct_fading=args.fading,
        color_restore=args.color_balance,
        sharpen=args.sharpen,
        backend=args.backend,
        ai_model_path=args.model,
        localized_only=not args.full_photo,
        debug_dir=args.steps_dir,
    )
    result.image.save(args.output)
    if result.saved_steps:
        print(f"Saved {len(result.saved_steps)} intermediate step images to {args.steps_dir}")
    print(f"Saved {args.output}; detected {result.mask_coverage:.2%} damage coverage; backend={result.backend_used}")
    if result.warning:
        print(f"Warning: {result.warning}")


if __name__ == "__main__":
    main()
