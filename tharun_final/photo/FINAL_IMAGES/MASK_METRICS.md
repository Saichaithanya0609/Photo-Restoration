# Mask and intermediate-image metrics

All arrays are saved at the same source-image resolution. `coverage` is the fraction of pixels with a value greater than zero.

| Step | Shape | Dtype | Non-zero | Coverage | Components | Largest component |
|---|---|---|---:|---:|---:|---:|
| `01_original_input` | `500x475x3` | `uint8` | — | 0.00% | — | — |
| `02_grayscale_luminance` | `500x475` | `uint8` | 237489 | 100.00% | — | — |
| `03_smooth` | `500x475` | `float32` | 237500 | 100.00% | — | — |
| `04_local_contrast` | `500x475` | `float32` | 110821 | 46.66% | — | — |
| `05a_symmetry_bright_response` | `500x475` | `float32` | 196565 | 82.76% | — | — |
| `05b_symmetry_candidates` | `500x475` | `uint8` | 5126 | 2.16% | 292 | 896 |
| `05c_symmetry_seeds` | `500x475` | `uint8` | 1965 | 0.83% | 37 | 718 |
| `05d_symmetry_hysteresis` | `500x475` | `uint8` | 3298 | 1.39% | 26 | 896 |
| `05e_symmetry_closed` | `500x475` | `uint8` | 4343 | 1.83% | — | — |
| `06a_frangi_vesselness` | `500x475` | `float32` | 186605 | 78.57% | — | — |
| `06b_frangi_binary` | `500x475` | `uint8` | 8798 | 3.70% | 460 | 633 |
| `06_frangi_ridges` | `500x475` | `uint8` | 6632 | 2.79% | — | — |
| `07a_combined_crack_candidates` | `500x475` | `uint8` | 8717 | 3.67% | 347 | 1409 |
| `07_crack_mask` | `500x475` | `uint8` | 9081 | 3.82% | 278 | 1502 |
| `08a_dust_response` | `500x475` | `float32` | 234152 | 98.59% | 4 | 234143 |
| `08b_dust_candidates` | `500x475` | `uint8` | 4532 | 1.91% | 335 | 1080 |
| `08_dust_mask` | `500x475` | `uint8` | 929 | 0.39% | 154 | 44 |
| `09_damage_mask` | `500x475` | `uint8` | 9725 | 4.09% | 316 | 1502 |
| `10_damage_overlay_red` | `500x475x3` | `uint8` | — | 0.00% | — | — |
