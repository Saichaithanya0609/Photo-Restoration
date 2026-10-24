
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from pipeline import process_image

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


app = FastAPI(
    title="Photo Restoration API",
    description="API for photo restoration",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Photo Restoration API is running"
    }


@app.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    denoise: bool = Form(False),
):
    # 1. Validate the uploaded file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPEG, PNG, or WebP image.",
        )

    # 2. Read the uploaded file
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # 3. Decode the bytes into an OpenCV image
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as an image.",
        )

    try:
        # 4. Run our pipeline
        result = process_image(image, denoise_enabled=denoise)

        # 5. Encode and save the result under Backend/outputs.
        success, encoded_image = cv2.imencode(".png", result)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Could not encode the restored image.",
            )

        png_bytes = encoded_image.tobytes()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_name = f"restored_{uuid4().hex}.png"
        (OUTPUT_DIR / output_name).write_bytes(png_bytes)

        # 6. Return the processed image
        return StreamingResponse(
            BytesIO(png_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="restored.png"',
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(exc)}",
        ) from exc
