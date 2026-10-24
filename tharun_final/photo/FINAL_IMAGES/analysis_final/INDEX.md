# Process Step Analysis

Intermediate step images captured during photo restoration.

| Step Name | Image File |
|---|---|
| `01_original_input` | [`01_original_input.png`](01_original_input.png) |
| `02_grayscale_luminance` | [`02_grayscale_luminance.png`](02_grayscale_luminance.png) |
| `03_smooth` | [`03_smooth.png`](03_smooth.png) |
| `04_local_contrast` | [`04_local_contrast.png`](04_local_contrast.png) |
| `05a_symmetry_bright_response` | [`05a_symmetry_bright_response.png`](05a_symmetry_bright_response.png) |
| `05b_symmetry_candidates` | [`05b_symmetry_candidates.png`](05b_symmetry_candidates.png) |
| `05b_symmetry_candidates_overlay` | [`05b_symmetry_candidates_overlay.png`](05b_symmetry_candidates_overlay.png) |
| `05c_symmetry_seeds` | [`05c_symmetry_seeds.png`](05c_symmetry_seeds.png) |
| `05c_symmetry_seeds_overlay` | [`05c_symmetry_seeds_overlay.png`](05c_symmetry_seeds_overlay.png) |
| `05d_symmetry_hysteresis` | [`05d_symmetry_hysteresis.png`](05d_symmetry_hysteresis.png) |
| `05d_symmetry_hysteresis_overlay` | [`05d_symmetry_hysteresis_overlay.png`](05d_symmetry_hysteresis_overlay.png) |
| `05e_symmetry_closed` | [`05e_symmetry_closed.png`](05e_symmetry_closed.png) |
| `06_frangi_ridges` | [`06_frangi_ridges.png`](06_frangi_ridges.png) |
| `06a_frangi_vesselness` | [`06a_frangi_vesselness.png`](06a_frangi_vesselness.png) |
| `06b_frangi_binary` | [`06b_frangi_binary.png`](06b_frangi_binary.png) |
| `06b_frangi_binary_overlay` | [`06b_frangi_binary_overlay.png`](06b_frangi_binary_overlay.png) |
| `07_crack_mask` | [`07_crack_mask.png`](07_crack_mask.png) |
| `07_crack_mask_overlay` | [`07_crack_mask_overlay.png`](07_crack_mask_overlay.png) |
| `07a_combined_crack_candidates` | [`07a_combined_crack_candidates.png`](07a_combined_crack_candidates.png) |
| `07a_combined_crack_candidates_overlay` | [`07a_combined_crack_candidates_overlay.png`](07a_combined_crack_candidates_overlay.png) |
| `08_dust_mask` | [`08_dust_mask.png`](08_dust_mask.png) |
| `08_dust_mask_overlay` | [`08_dust_mask_overlay.png`](08_dust_mask_overlay.png) |
| `08_dust_mask_raw` | [`08_dust_mask_raw.png`](08_dust_mask_raw.png) |
| `08_dust_mask_raw_overlay` | [`08_dust_mask_raw_overlay.png`](08_dust_mask_raw_overlay.png) |
| `08a_dust_response_raw` | [`08a_dust_response_raw.png`](08a_dust_response_raw.png) |
| `08b_dust_response_thresholded` | [`08b_dust_response_thresholded.png`](08b_dust_response_thresholded.png) |
| `08c_dust_candidates` | [`08c_dust_candidates.png`](08c_dust_candidates.png) |
| `08c_dust_candidates_overlay` | [`08c_dust_candidates_overlay.png`](08c_dust_candidates_overlay.png) |
| `09_damage_mask` | [`09_damage_mask.png`](09_damage_mask.png) |
| `09_damage_mask_overlay` | [`09_damage_mask_overlay.png`](09_damage_mask_overlay.png) |
| `10_damage_overlay_red` | [`10_damage_overlay_red.png`](10_damage_overlay_red.png) |
| `11_inpainting_output_raw` | [`11_inpainting_output_raw.png`](11_inpainting_output_raw.png) |
| `11a_grain_reduction_denoised` | [`11a_grain_reduction_denoised.png`](11a_grain_reduction_denoised.png) |
| `12_final_restored_composite` | [`12_final_restored_composite.png`](12_final_restored_composite.png) |

## Auditing the masks

- `MASK_METRICS.md` / `MASK_METRICS.json` report shape, pixel count, coverage, and connected components for every detection stage.
- Mask stages also have a corresponding `_overlay.png` showing the stage in red over the original image.
- `RESTORATION_METRICS.md` / `RESTORATION_METRICS.json` report the inpainting and final-composite stages.
