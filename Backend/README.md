# Photo Restoration Backend

The FastAPI service accepts JPEG, PNG, and WebP uploads at `POST /restore` and
returns the restored result as a PNG. The React editor can optionally request
Non-Local Means (NLM) denoising; it is off by default.

## Processing flow

1. `main.py` validates and decodes the upload into an OpenCV BGR image.
2. `pipeline.py` optionally calls `image_processor.py` for NLM denoising.
3. The root pipeline converts BGR to PIL RGB and calls
   `Restoration.pipeline.restore_image()` for damage detection, mask refinement,
   and inpainting.
4. The result is converted back to BGR, encoded as PNG, saved with a unique
   filename in `outputs/`, and returned to the browser. No contrast
   enhancement is applied in this flow.

## Run locally

From this directory, create and activate a virtual environment, then install
the backend requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive API docs are at
`http://127.0.0.1:8000/docs`.

Each successful restoration is saved as `outputs/restored_<id>.png`. These
generated files are local artifacts and are not committed to Git.
