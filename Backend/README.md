# Photo Restoration Backend

FastAPI accepts an uploaded photograph, optionally denoises it, runs crack
detection and mask-based inpainting, saves the restored PNG locally, and
returns that PNG to the frontend.

## Requirements and installation

From the repository root in PowerShell:

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

## Start the API

From the `Backend` directory with the virtual environment active:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API base: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health route: `GET /`

The frontend development server runs at `http://localhost:5173`. These local
frontend origins are allowed by the API's CORS configuration:
`http://localhost:5173` and `http://127.0.0.1:5173`.

## API endpoint

### `POST /restore`

Accepts `multipart/form-data`:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | JPEG, PNG, or WebP file | Yes | Photograph to restore |
| `remove_noise` | Boolean | No | Run NLM denoising before detection; defaults to `false` |

On success, returns the processed image as `image/png`. The API also writes a
unique copy named `restored_<id>.png` into `Backend/outputs/`. Generated files
are ignored by Git; `.gitkeep` preserves the empty folder in a fresh checkout.

Possible validation errors include `400` for empty or undecodable files and
`415` for unsupported upload types. Processing or PNG-encoding errors return
`500` with a detail message.

## Manual API check

With the server running, open another terminal in the `Backend` directory and
run this PowerShell command:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/restore" `
  -F "file=@test_images/image02.jpeg;type=image/jpeg" `
  -F "remove_noise=false" `
  --output outputs/manual_test.png
```

Open `outputs/manual_test.png` to inspect the API response. The server also
stores its own uniquely named copy in the same folder. To check denoising,
repeat the request with `remove_noise=true`. The same request can be sent from
Swagger UI at `/docs`.

There are no automated backend test scripts in the current project. The
commands above exercise the live upload endpoint; `Backend/test_images/`
contains local sample images.

## Processing flow

1. `main.py` validates and decodes the uploaded image to an OpenCV BGR array.
2. `pipeline.py` optionally runs NLM denoising through `image_processor.py`.
3. The root pipeline converts BGR to PIL RGB and calls
   `Restoration.pipeline.restore_image()`.
4. The `Restoration` package detects candidate damage, refines the mask, and
   inpaints masked regions.
5. `main.py` encodes and saves the result, then returns the PNG response.

No contrast enhancement is applied in the active flow.
