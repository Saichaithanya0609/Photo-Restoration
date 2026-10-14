
import cv2


def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if image is None:
        raise ValueError("Input image was not received.")

    # Grayscale images need a different path.
    if image.ndim == 2:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        return clahe.apply(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or BGR image.")

    # Convert BGR to LAB so we can adjust luminance.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    improved_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge((improved_lightness, a, b))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
