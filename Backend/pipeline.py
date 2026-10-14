
import cv2

from image_repair import repair_image
from enhancement import enhance_contrast


def process_image(
    image,
    noise_removal_fn=None,
    scratch_detection_fn=None,
    repair_method="telea",
    apply_contrast=True,
):
    """
    Coordinate the photo restoration pipeline.

    Teammate algorithms are passed in as functions so their
    implementations can be integrated when they are ready.
    """

    if image is None:
        raise ValueError("Input image was not received.")

    # Stage 1: Noise removal
    if noise_removal_fn is not None:
        denoised = noise_removal_fn(image)
    else:
        # Temporary fallback until the teammate's module is ready.
        denoised = image.copy()

    # Stage 2: Scratch detection
    if scratch_detection_fn is not None:
        damage_mask = scratch_detection_fn(denoised)

        if damage_mask is None:
            raise ValueError("Scratch detection returned no mask.")

        if damage_mask.ndim == 3:
            damage_mask = cv2.cvtColor(
                damage_mask, cv2.COLOR_BGR2GRAY
            )

        if damage_mask.shape[:2] != denoised.shape[:2]:
            raise ValueError(
                "Damage mask dimensions do not match the image."
            )

        # Stage 3: Scratch repair
        repaired = repair_image(
            denoised,
            damage_mask,
            method=repair_method,
        )
    else:
        # Temporary fallback until the detector is ready.
        repaired = denoised.copy()

    # Stage 4: Contrast enhancement
    if apply_contrast:
        result = enhance_contrast(repaired)
    else:
        result = repaired

    return result
