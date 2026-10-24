from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from restoration import (
    detect_cracks,
    detect_dust,
    detect_damage,
    inpaint_high_quality,
    restore_from_mask,
    restore_image,
)


class RestorationTests(unittest.TestCase):

    # ── Detection tests ────────────────────────────────────────────────────────

    def test_clean_image_has_no_large_mask(self) -> None:
        image = np.full((160, 220), 100, dtype=np.uint8)
        cv2.circle(image, (110, 80), 35, 210, -1)  # broad bright object — not a crack
        mask = detect_cracks(image, sensitivity=0.8)
        self.assertLess(np.count_nonzero(mask) / mask.size, 0.02)

    def test_thin_white_crack_is_detected(self) -> None:
        image = np.full((160, 220), 80, dtype=np.uint8)
        truth = np.zeros_like(image)
        cv2.line(image, (15, 25), (205, 130), 245, 1)
        cv2.line(truth, (15, 25), (205, 130), 255, 1)
        mask = detect_cracks(image, sensitivity=1.2)
        overlap = np.count_nonzero((mask > 0) & (truth > 0))
        predicted = np.count_nonzero(mask)
        actual = np.count_nonzero(truth)
        recall = overlap / max(1, actual)
        precision = overlap / max(1, predicted)
        iou = overlap / max(1, predicted + actual - overlap)
        self.assertGreater(recall, 0.35)
        self.assertGreater(precision, 0.25)
        self.assertGreater(iou, 0.15)
        self.assertLess(np.count_nonzero(mask) / mask.size, 0.08)

    def test_broad_white_region_is_not_flagged(self) -> None:
        image = np.full((160, 220), 70, dtype=np.uint8)
        image[35:125, 55:165] = 245
        mask = detect_cracks(image, sensitivity=1.2)
        inside = np.count_nonzero(mask[35:125, 55:165]) / (90 * 110)
        self.assertLess(inside, 0.08)

    def test_dark_and_intersecting_cracks_are_supported(self) -> None:
        image = np.full((256, 256), 180, dtype=np.uint8)
        truth = np.zeros_like(image)
        cv2.line(image, (20, 220), (232, 34), 30, 2)
        cv2.line(truth, (20, 220), (232, 34), 255, 2)
        cv2.line(image, (34, 45), (220, 180), 35, 5)
        cv2.line(truth, (34, 45), (220, 180), 255, 5)
        mask = detect_cracks(image, sensitivity=1.2, detect_dark=True)
        overlap = np.count_nonzero((mask > 0) & (truth > 0))
        self.assertGreater(overlap / max(1, np.count_nonzero(truth)), 0.40)
        self.assertLess(np.count_nonzero(mask) / mask.size, 0.12)

    def test_fragmented_crack_is_reconnected(self) -> None:
        image = np.full((180, 240), 75, dtype=np.uint8)
        truth = np.zeros_like(image)
        for start, end in [((15, 25), (70, 65)), ((75, 68), (135, 110)), ((140, 114), (220, 155))]:
            cv2.line(image, start, end, 245, 2)
            cv2.line(truth, start, end, 255, 2)
        mask = detect_cracks(image, sensitivity=1.5)
        overlap = np.count_nonzero((mask > 0) & (truth > 0))
        self.assertGreater(overlap / max(1, np.count_nonzero(truth)), 0.35)

    def test_dust_detector_is_compact(self) -> None:
        image = np.full((180, 240), 90, dtype=np.uint8)
        cv2.circle(image, (50, 50), 3, 245, -1)
        cv2.circle(image, (180, 130), 2, 10, -1)
        mask = detect_dust(image, sensitivity=1.0)
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertLess(np.count_nonzero(mask) / mask.size, 0.03)

    def test_dust_mask_never_changes_image_dimensions(self) -> None:
        image = np.full((96, 128), 90, dtype=np.uint8)
        cv2.circle(image, (35, 35), 3, 245, -1)
        mask = detect_dust(image, sensitivity=1.0)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertLessEqual(np.count_nonzero(mask), image.size)

    def test_frangi_detail_is_preserved_in_crack_mask(self) -> None:
        image = np.full((160, 220), 80, dtype=np.uint8)
        cv2.line(image, (15, 25), (205, 130), 245, 1)
        steps: dict[str, np.ndarray] = {}
        refined = detect_cracks(image, sensitivity=1.2, steps_dict=steps)
        frangi = steps["06_frangi_ridges"]
        self.assertTrue(np.all(refined[frangi > 0] > 0))

    def test_dust_survives_crack_mask_union(self) -> None:
        """Dust is unioned after crack refinement, so round spots are retained."""
        gray = np.full((180, 240), 90, dtype=np.uint8)
        cv2.circle(gray, (50, 50), 3, 245, -1)
        image = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        detection = detect_damage(image, bright_cracks=False, detect_dust_marks=True)
        self.assertGreater(np.count_nonzero(detection.mask_np), 0)

    def test_mask_expansion_is_not_applied(self) -> None:
        """Legacy expansion arguments must not dilate the unified mask."""
        rgb = np.full((120, 180, 3), 90, dtype=np.uint8)
        cv2.line(rgb, (15, 20), (165, 90), (245, 245, 245), 2)
        image = Image.fromarray(rgb)
        without_expansion = detect_damage(image, sensitivity=1.2, expansion=0)
        with_legacy_expansion = detect_damage(image, sensitivity=1.2, expansion=8)
        self.assertTrue(np.array_equal(without_expansion.mask_np, with_legacy_expansion.mask_np))

    # ── Pipeline / two-stage tests ─────────────────────────────────────────────

    def test_empty_detection_returns_copy_unchanged(self) -> None:
        original = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8), mode="RGB")
        result = restore_image(original)
        self.assertFalse(result.restoration_applied)
        self.assertTrue(np.array_equal(np.asarray(original), np.asarray(result.image)))

    def test_detect_damage_returns_detection_result(self) -> None:
        rgb = np.full((120, 180, 3), 90, dtype=np.uint8)
        cv2.line(rgb, (15, 20), (165, 90), (245, 245, 245), 2)
        image = Image.fromarray(rgb)
        detection = detect_damage(image, sensitivity=1.2)
        self.assertEqual(detection.mask_np.shape, (120, 180))
        self.assertIsNotNone(detection.mask_image)
        self.assertGreaterEqual(detection.coverage, 0.0)
        self.assertLessEqual(detection.coverage, 1.0)

    def test_restore_from_mask_applies_inpainting(self) -> None:
        rgb = np.full((120, 180, 3), 90, dtype=np.uint8)
        cv2.line(rgb, (15, 20), (165, 90), (245, 245, 245), 2)
        image = Image.fromarray(rgb)
        detection = detect_damage(image, sensitivity=1.2)
        if detection.coverage > 0:
            result = restore_from_mask(image, detection, denoise_strength=0)
            self.assertTrue(result.restoration_applied)
            self.assertIsNotNone(result.image)

    def test_high_quality_backend_name_matches_implementation(self) -> None:
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        image[15, 15] = 240
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[15, 15] = 255
        _, backend = inpaint_high_quality(image, mask)
        self.assertIn(backend, {"biharmonic", "Navier–Stokes"})

    def test_output_preserves_pixels_outside_mask(self) -> None:
        rgb = np.full((120, 180, 3), 90, dtype=np.uint8)
        cv2.line(rgb, (15, 20), (165, 90), (245, 245, 245), 1)
        original = Image.fromarray(rgb, mode="RGB")
        detection = detect_damage(original, sensitivity=1.2)
        if detection.coverage == 0:
            self.skipTest("No damage detected on this synthetic image at this sensitivity")
        result = restore_from_mask(original, detection)
        before = np.asarray(original)
        after = np.asarray(result.image)
        mask = np.asarray(result.mask) > 0
        self.assertEqual(before.shape, after.shape)
        # Every pixel outside the mask must be exactly preserved
        self.assertTrue(np.array_equal(before[~mask], after[~mask]))

    def test_legacy_restore_image_accepts_old_kwargs(self) -> None:
        """Old keyword arguments must be silently absorbed without errors."""
        image = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8), mode="RGB")
        # These kwargs no longer exist in the new API; they must not raise
        result = restore_image(
            image,
            detection_sensitivity=1.0,
            mask_expansion=2.0,
            max_coverage=0.25,
            backend="lama",
            correct_fading=True,
            sharpen=True,
        )
        self.assertIsNotNone(result)

    def test_debug_dir_saves_step_images(self) -> None:
        """Ensure every step image and INDEX.md are properly written to debug_dir."""
        import tempfile
        import os

        rgb = np.full((120, 180, 3), 90, dtype=np.uint8)
        cv2.line(rgb, (15, 20), (165, 90), (245, 245, 245), 2)
        image = Image.fromarray(rgb)

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = restore_image(image, sensitivity=1.2, denoise_strength=2, debug_dir=tmp_dir)
            self.assertGreater(len(res.saved_steps), 8)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "01_original_input.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "09_damage_mask.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "12_final_restored_composite.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "INDEX.md")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "MASK_METRICS.md")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "MASK_METRICS.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "07_crack_mask_overlay.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "RESTORATION_METRICS.md")))

    def test_debug_dir_saves_dust_pipeline_stages(self) -> None:
        import os
        import tempfile

        gray = np.full((96, 128), 90, dtype=np.uint8)
        cv2.circle(gray, (35, 35), 3, 245, -1)
        image = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

        with tempfile.TemporaryDirectory() as tmp_dir:
            detection = detect_damage(
                image,
                bright_cracks=False,
                detect_dust_marks=True,
                debug_dir=tmp_dir,
            )
            self.assertEqual(detection.mask_np.shape, gray.shape)
            for filename in (
                "08a_dust_response_raw.png",
                "08b_dust_response_thresholded.png",
                "08c_dust_candidates.png",
                "08_dust_mask_raw.png",
                "08_dust_mask.png",
                "08_dust_mask_overlay.png",
                "MASK_METRICS.md",
                "MASK_METRICS.json",
            ):
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, filename)))


if __name__ == "__main__":
    unittest.main()
