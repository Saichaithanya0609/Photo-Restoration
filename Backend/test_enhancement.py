
import cv2
from enhancement import enhance_contrast

# Load a test photo
image = cv2.imread("./images/repaired_ns_blend.png")

if image is None:
    raise FileNotFoundError(
        "Could not load ./images/image02.jpeg"
    )

# Apply contrast enhancement
enhanced = enhance_contrast(image)

# Save the output
output_path = "./images/enhanced_contrast.png"
success = cv2.imwrite(output_path, enhanced)

if not success:
    raise OSError("Could not save enhanced image.")

print("Contrast enhancement completed!")
print("Original shape:", image.shape)
print("Enhanced shape:", enhanced.shape)
print("Saved to:", output_path)
