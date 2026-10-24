import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import pipeline


class BackendPipelineIntegrationTests(unittest.TestCase):
    def test_selected_denoising_runs_before_restoration(self):
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        source[:, :, 0] = 15
        source[:, :, 2] = 230
        denoised = source.copy()
        restored_rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        restored_rgb[:, :, 0] = 200
        events = []

        def fake_denoise(image, method, strength):
            events.append(("denoise", image, method, strength))
            return denoised

        def fake_restore(image, *, sensitivity):
            events.append(("restore", image, sensitivity))
            self.assertIsInstance(image, Image.Image)
            np.testing.assert_array_equal(np.asarray(image), denoised[:, :, ::-1])
            return type("Restoration", (), {"image": Image.fromarray(restored_rgb)})()

        with (
            patch.object(pipeline, "denoise_image", side_effect=fake_denoise),
            patch.object(pipeline, "restore_image", side_effect=fake_restore),
        ):
            result = pipeline.process_image(source, denoise_enabled=True)

        self.assertEqual([event[0] for event in events], ["denoise", "restore"])
        self.assertEqual(events[0][2:], ("nlm", 25))
        self.assertEqual(events[1][2], 1.0)
        np.testing.assert_array_equal(result, restored_rgb[:, :, ::-1])

    def test_denoising_is_skipped_by_default(self):
        source = np.zeros((8, 10, 3), dtype=np.uint8)
        source[:, :, 0] = 17
        source[:, :, 2] = 211
        restored_rgb = source[:, :, ::-1].copy()

        def fake_restore(image, *, sensitivity):
            self.assertIsInstance(image, Image.Image)
            np.testing.assert_array_equal(np.asarray(image), restored_rgb)
            return type("Restoration", (), {"image": Image.fromarray(restored_rgb)})()

        with (
            patch.object(pipeline, "denoise_image") as denoise,
            patch.object(pipeline, "restore_image", side_effect=fake_restore),
        ):
            result = pipeline.process_image(source)

        denoise.assert_not_called()
        np.testing.assert_array_equal(result, source)

if __name__ == "__main__":
    unittest.main()
