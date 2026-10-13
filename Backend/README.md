# Photo Restoration — Image Denoising Module

## Included algorithms
- Median filtering: useful for salt-and-pepper noise.
- Gaussian filtering: simple smoothing baseline.
- Non-Local Means (NLM): patch-similarity denoising for photographic noise.

## Setup
From this `Backend` directory:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place a test photo at `test_images/old_photo.jpg`, then run:
```powershell
python test_denoising.py
```

The individual PNG outputs and `comparison.png` are saved under `outputs/`.

## Notes
- The original image is not modified.
- Strength is a method-specific control from 0 to 100, not a calibrated noise estimate.
- PSNR/SSIM are meaningful as quality metrics when a clean reference image is available.
- This module is standalone; connect it to the team's Flask API only after coordinating the API contract.
