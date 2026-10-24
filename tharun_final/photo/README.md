# Photo Restore

Local Streamlit photo restoration focused on **damage localization first**.
The default path detects crack-like structures, removes them only inside the
displayed mask, and leaves every other pixel byte-for-byte unchanged.

## Pipeline

1. Preserve the original RGB image and derive a grayscale detection reference.
2. Detect bright cracks using multi-scale, multi-angle, two-sided local contrast.
3. Optionally detect dark cracks and compact dust/blemishes.
4. Join justified fragmented/diagonal/intersecting candidates and reject broad
   blobs using connected-component shape, elongation, thickness, and contrast.
5. Refine crack candidates once, then union compact dust directly with the
   refined crack mask. No broad seam expansion is applied.
6. Use biharmonic inpainting when scikit-image is available, otherwise report
   and use OpenCV Navier–Stokes honestly.
7. Optionally apply localized grain reduction only inside repaired regions.

This is not a face-generation system: missing information cannot always be
recovered faithfully, and optional AI inpainting can invent plausible content.
Inspect the mask and compare the result before downloading it.

## Run the classical app

```bash
cd /Users/tharun/Desktop/photo1/photo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a photograph, inspect **Original / Exact restoration mask / Restored**,
and download the restored PNG. The safety limit stops automatic inpainting when
the mask is unexpectedly broad.

## Unified engine and process images

All restoration logic is in `restoration.py`. The helper below writes the
original image, exact mask, restored image, and intermediate process images to
`FINAL_IMAGES/` for inspection:

```bash
python3 run_step_analysis.py images_and_steps/images\ \(1\).jpeg \
  --output-dir FINAL_IMAGES --dust --denoise 3
```

For mask debugging, inspect the generated `06_frangi_ridges.png`,
`07a_combined_crack_candidates.png`, `07_crack_mask.png`, and the four dust
stages beginning with `08a_dust_response_raw.png`. `08_dust_mask.png` is the
dust-only mask after crack overlap is removed; the final damage mask still
uses a direct bitwise OR. Each binary mask also has a
`_overlay.png` version showing that stage in red over the original. The
`MASK_METRICS.md` and `MASK_METRICS.json` files prove that every stage keeps
the original image dimensions and quantify its coverage. The script also writes
easy-to-find aliases: `original_input.png`, `final_damage_mask.png`, and
`restored_output.png`.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q app.py restoration.py run_step_analysis.py tests
```

The tests cover clean images, bright/dark/wide/intersecting cracks, fragmented
damage, broad bright regions, dust, large-mask safety, backend reporting, and
exact pixel preservation outside the localized mask.
