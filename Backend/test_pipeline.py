
import cv2

from pipeline import process_image


image = cv2.imread("./images/image02.jpeg")

if image is None:
    raise FileNotFoundError(
        "Could not load ./images/image02.jpeg"
    )

result = process_image(image)

output_path = "./images/pipeline_test.png"

if not cv2.imwrite(output_path, result):
    raise OSError("Could not save pipeline output.")

print("Pipeline test completed!")
print("Input shape:", image.shape)
print("Output shape:", result.shape)
print("Saved to:", output_path)
