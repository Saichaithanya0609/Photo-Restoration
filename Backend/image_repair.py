
import cv2
import numpy as np


def repair_image(image, damage_mask, method="telea", radius=None):
    """
    Repair damaged regions in an image using OpenCV inpainting.

    Args:
        image: Input grayscale or color image.
        damage_mask: Single-channel mask; nonzero pixels indicate
                     regions to repair.
        method: "telea", "ns", or "ns_blend".
        radius: Inpainting neighborhood radius. Uses a method-specific
                default when omitted.

    Returns:
        Repaired image as a NumPy array.
    """

    # 1. Validate inputs
    if image is None:
        raise ValueError("Input image was not received.")

    if damage_mask is None:
        raise ValueError("Damage mask was not received.")

    # 2. Convert a color mask to grayscale
    if damage_mask.ndim == 3:
        damage_mask = cv2.cvtColor(
            damage_mask, cv2.COLOR_BGR2GRAY
        )

    # 3. OpenCV inpainting requires an 8-bit mask
    if damage_mask.dtype != np.uint8:
        damage_mask = np.clip(
            damage_mask, 0, 255
        ).astype(np.uint8)

    # 4. Ensure image and mask dimensions match
    if image.shape[:2] != damage_mask.shape[:2]:
        raise ValueError(
            "Image and damage mask dimensions do not match."
        )

    # 5. Select the inpainting algorithm
    if method == "telea":
        flag = cv2.INPAINT_TELEA
        default_radius = 3

    elif method in ("ns", "ns_blend"):
        flag = cv2.INPAINT_NS
        default_radius = 7

    else:
        raise ValueError(
            "Method must be 'telea', 'ns', or 'ns_blend'."
        )

    if radius is None:
        radius = default_radius

    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("Radius must be a positive number.")

    # 6. Repair the masked regions
    repaired = cv2.inpaint(
        image, damage_mask, float(radius), flag
    )

    # 7. Optional post-denoising for comparison
    if method == "ns_blend":
        if repaired.ndim == 2:
            repaired = cv2.fastNlMeansDenoising(
                repaired, None, 5, 7, 21
            )
        else:
            repaired = cv2.fastNlMeansDenoisingColored(
                repaired, None, 5, 5, 7, 21
            )

    return repaired


if __name__ == "__main__":
    image = cv2.imread("./images/image02.jpeg")

    damage_mask = cv2.imread(
        "./images/detected_mask.png",
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            "Could not load ./images/image02.jpeg"
        )

    if damage_mask is None:
        raise FileNotFoundError(
            "Could not load ./images/detected_mask.png"
        )

    for method in ("telea", "ns", "ns_blend"):
        result = repair_image(
            image, damage_mask, method=method
        )

        output_path = f"./images/repaired_{method}.png"

        if not cv2.imwrite(output_path, result):
            raise OSError(f"Could not save {output_path}")

        print(f"Saved: {output_path}")
