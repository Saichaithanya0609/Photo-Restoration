# Photo Restoration Detection Process

The restoration engine now lives in one module: `restoration.py`. The
Streamlit app, tests, and `run_step_analysis.py` all call the same public API.

## Detection flow

1. Convert the uploaded image to BGR and derive an 8-bit luminance reference.
2. Detect bright cracks with multi-scale, multi-angle two-sided local contrast.
3. Optionally detect dark cracks with the corresponding valley response.
4. Reuse the 60 shifted flank arrays during the response and continuity passes.
5. Combine symmetry evidence with a gated multi-scale Frangi ridge response.
6. Run `refine_mask()` once on crack candidates to reject broad blobs and weak
   non-line structures using connected-component geometry and local contrast.
7. Optionally detect compact dust/blemish spots separately.
8. Combine the refined crack mask and dust mask with `cv2.bitwise_or`.

Dust is intentionally added after crack refinement. Round dust spots should not
be sent through the crack elongation filter, whose purpose is to reject
non-line-shaped crack candidates.

## No mask bloat

The final mask is the exact union of the refined crack mask and the dust mask.
There is no second `refine_mask()` call and no fixed seam expansion or
aggressive final dilation. This keeps the inpainting region tight and protects
undamaged detail around a defect.

The only dilation still used inside crack refinement creates a local ring for
measuring surrounding contrast; it is not applied to the returned mask.

## Why fine crack details were disappearing

The Frangi stage is a pixel-level ridge detector, so it can correctly show
thin branches and fragmented parts of a crack. The old `refine_mask()` then
treated the union of Frangi and symmetry candidates as ordinary connected
components. Its minimum-area, minimum-dimension, elongation, and thickness
rules rejected many small Frangi fragments even when the Frangi response was
already thresholded and flank-gated. That made `07_crack_mask.png` look much
thinner than `06_frangi_ridges.png`.

The unified code still runs one crack refinement pass, but it passes the
high-confidence, flank-gated Frangi mask as `preserve_mask`. Fine Frangi
pixels are therefore retained while the normal geometry filter continues to
clean the symmetry candidates. The saved `06_frangi_ridges.png`,
`07a_combined_crack_candidates.png`, and `07_crack_mask.png` make this change
easy to compare.

## Why the dust mask looked too large

The morphological dust response is not itself a mask. Top-hat and black-hat
responses can be positive across much of a textured photograph, so a raw
response visualization may look almost full-frame even though the final dust
mask is sparse. The output now labels these separately:

- `08a_dust_response_raw.png`: diagnostic response map, not a mask.
- `08b_dust_response_thresholded.png`: response after the numeric threshold.
- `08c_dust_candidates.png`: binary candidates before component filtering.
- `08_dust_mask_raw.png`: compact dust candidates before removing pixels that
  are already covered by the crack mask.
- `08_dust_mask.png`: final compact dust-only mask used by restoration.

`MASK_METRICS.md` and `MASK_METRICS.json` record each stage's exact shape,
non-zero pixel count, coverage, connected-component count, and largest
component. Every stage is asserted to have the original image dimensions;
there is no valid path for a mask to exceed the image array.

## Restoration flow

`restore_from_mask()` selects a classical backend based on mask coverage:

- Small masks use OpenCV Navier–Stokes for a fast localized repair.
- Larger masks use scikit-image biharmonic inpainting when available.
- If scikit-image is unavailable, the function falls back to OpenCV
  Navier–Stokes and reports `Navier–Stokes`, not `biharmonic`.

The restored image is composited over a copy of the original only where the
mask is active. Pixels outside the mask are preserved exactly. Optional grain
reduction is also applied only inside the repaired region.

## Process images

Run:

```bash
python3 run_step_analysis.py 'images_and_steps/images (1).jpeg' \
  --output-dir FINAL_IMAGES --dust --denoise 3
```

`FINAL_IMAGES/` contains the original, luminance, detector stages, exact
damage mask, red overlay, raw inpainting output, and final restored composite.
The generated `INDEX.md` lists each numbered image.

## Public API

```python
from restoration import DetectionResult, detect_damage, restore_from_mask, restore_image
```

The legacy package modules, colourization adapter, LaMa adapter, and broken
CLI were removed so there is one source of truth for restoration behavior.
