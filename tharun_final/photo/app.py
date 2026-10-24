import io
import os
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

from restoration import DetectionResult, detect_damage, restore_from_mask, restore_image


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Photo Restore", page_icon="🖼️", layout="wide")
st.title("🖼️ Photo Restore")
st.caption("Detect cracks and damage in old photographs, inspect the mask, then restore.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_red_overlay(original: Image.Image, mask: Image.Image) -> Image.Image:
    """Blend a semi-transparent red highlight over the detected damage regions."""
    base = original.convert("RGBA")
    red_layer = Image.new("RGBA", original.size, (220, 40, 40, 0))
    alpha = mask.point(lambda p: 165 if p > 0 else 0)
    red_layer.putalpha(alpha)
    return Image.alpha_composite(base, red_layer).convert("RGB")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Damage type")
    bright_cracks = st.checkbox("Bright cracks & scratches", value=True)
    dark_cracks = st.checkbox(
        "Dark fracture lines",
        value=False,
        help=(
            "Enable for dark crack lines on a light background.\n\n"
            "Keep off for most photographs — dark hair and eyelashes can "
            "trigger this mode."
        ),
    )
    dust = st.checkbox(
        "Dust and small blemishes",
        value=False,
        help="Detect small compact spots and dust marks separately from line cracks.",
    )

    st.divider()

    st.header("Mask sensitivity")
    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.4,
        max_value=1.8,
        value=1.0,
        step=0.1,
        help=(
            "Controls how aggressively damage is included in the mask.\n\n"
            "• **Raise** if cracks are being missed.\n"
            "• **Lower** if hair, eyelashes, or texture bleeds into the mask.\n\n"
            "Re-click *Detect damage* after adjusting."
        ),
    )

    st.divider()

    st.header("Enhancement")
    denoise = st.slider(
        "Grain reduction",
        min_value=0,
        max_value=10,
        value=0,
        help="Reduce film grain inside repaired areas. 0 = off.",
    )

    st.divider()

    st.header("Step Analysis")
    save_steps = st.checkbox(
        "Save all process step images",
        value=True,
        help="Saves an image after every transformation and filtering step into a folder for detailed inspection.",
    )
    steps_dir = st.text_input("Folder to save step images", value="FINAL_IMAGES")
    st.caption(
        "Each detector stage is saved at the original image size. Binary masks "
        "also get red overlays; MASK_METRICS.md reports exact coverage and dimensions."
    )


# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an old photograph here, or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
)

if uploaded is None:
    st.markdown(
        "### How to restore a photo\n"
        "1. **Upload** your photograph above.\n"
        "2. Click **Detect damage** — cracks and scratches are highlighted in red.\n"
        "3. Inspect the overlay. Adjust *Mask sensitivity* and re-detect until the "
        "mask looks right.\n"
        "4. Click **Restore photo** to fill the detected regions.\n"
        "5. **Download** the restored image."
    )
else:
    # Reset session state when a new file is uploaded
    file_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("_file_id") != file_id:
        for key in ("detection", "restore_result"):
            st.session_state.pop(key, None)
        st.session_state["_file_id"] = file_id

    try:
        original = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        st.stop()

    # ── Action buttons ────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        detect_clicked = st.button(
            "🔍 Detect damage",
            use_container_width=True,
            type="primary",
        )

    with col_b:
        have_detection = (
            "detection" in st.session_state
            and st.session_state["detection"].coverage > 0
        )
        restore_clicked = st.button(
            "✨ Restore photo",
            use_container_width=True,
            type="secondary",
            disabled=not have_detection,
            help="Run detection first, then restore." if not have_detection else None,
        )

    # ── Stage 1: Detect ───────────────────────────────────────────────────────
    active_steps_dir = steps_dir.strip() if save_steps and steps_dir.strip() else None

    if detect_clicked:
        with st.spinner("Analysing photograph for damage…"):
            detection = detect_damage(
                original,
                sensitivity=sensitivity,
                bright_cracks=bright_cracks,
                dark_cracks=dark_cracks,
                detect_dust_marks=dust,
                debug_dir=active_steps_dir,
            )
        st.session_state["detection"] = detection
        # Clear any previous restore result — mask may have changed
        st.session_state.pop("restore_result", None)

    # ── Stage 2: Restore ──────────────────────────────────────────────────────
    if restore_clicked and "detection" in st.session_state:
        detection: DetectionResult = st.session_state["detection"]
        if detection.coverage == 0:
            st.warning("No damage detected. There is nothing to restore.")
        else:
            with st.spinner(
                "Running high-quality inpainting — "
                "this may take 10–30 seconds for larger images…"
            ):
                result = restore_from_mask(
                    original,
                    detection,
                    denoise_strength=denoise,
                    debug_dir=active_steps_dir,
                )
            st.session_state["restore_result"] = result

    # ── Display ───────────────────────────────────────────────────────────────
    detection: DetectionResult | None = st.session_state.get("detection")
    result = st.session_state.get("restore_result")

    if detection is None:
        # Nothing detected yet — show the uploaded photo
        st.image(original, caption="Uploaded photo", use_container_width=True)

    elif result is None:
        # Detection complete, not yet restored — show red overlay preview
        overlay = _make_red_overlay(original, detection.mask_image)
        cols = st.columns(2)
        with cols[0]:
            st.image(original, caption="Original", use_container_width=True)
        with cols[1]:
            pct = f"{detection.coverage:.2%}"
            st.image(
                overlay,
                caption=f"Detected damage ({pct} of image) — red areas will be repaired",
                use_container_width=True,
            )

        if detection.warning:
            if detection.coverage == 0:
                st.info(detection.warning)
            else:
                st.warning(detection.warning)
        elif detection.coverage > 0:
            st.success(
                f"Found damage covering **{pct}** of the photo. "
                "Adjust the sensitivity slider and re-detect if needed, "
                "then click **Restore photo**."
            )

    else:
        # Both stages done — show original, overlay, restored side by side
        overlay = _make_red_overlay(original, detection.mask_image)
        cols = st.columns(3)
        with cols[0]:
            st.image(original, caption="Original", use_container_width=True)
        with cols[1]:
            st.image(
                overlay,
                caption=f"Damage mask ({detection.coverage:.2%})",
                use_container_width=True,
            )
        with cols[2]:
            st.image(
                result.image,
                caption=f"Restored  ·  {result.backend_used}",
                use_container_width=True,
            )

        if result.warning:
            st.warning(result.warning)
        else:
            st.success(
                f"Restoration complete using **{result.backend_used}** inpainting."
            )

        # Download button
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download restored image",
            data=buf.getvalue(),
            file_name="restored_photo.png",
            mime="image/png",
            use_container_width=True,
        )

        with st.expander("Pipeline steps"):
            for step in result.steps:
                st.write(f"• {step}")

    # ── Step Analysis Explorer ────────────────────────────────────────────────
    steps_to_show = {}
    if result is not None and result.saved_steps:
        steps_to_show = result.saved_steps
    elif detection is not None and detection.saved_steps:
        steps_to_show = detection.saved_steps

    if steps_to_show:
        with st.expander("🔬 Step-by-Step Intermediate Analysis (Click to inspect each step)", expanded=False):
            st.caption(f"📁 All **{len(steps_to_show)}** intermediate step images have been saved to `{steps_dir}/`")
            step_keys = sorted(steps_to_show.keys())
            selected_step = st.selectbox(
                "Select a processing step to inspect:",
                step_keys,
                index=len(step_keys) - 1,
                help="Inspect the exact image produced at this transformation step.",
            )
            if selected_step and selected_step in steps_to_show:
                step_file = steps_to_show[selected_step]
                if os.path.exists(step_file):
                    st.image(step_file, caption=f"Step image: {selected_step}", use_container_width=True)

            # Zip download of all intermediate steps
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for s_name, s_path in steps_to_show.items():
                    if os.path.exists(s_path):
                        zf.write(s_path, arcname=os.path.basename(s_path))
                for aux_name in (
                    "INDEX.md",
                    "MASK_METRICS.md",
                    "MASK_METRICS.json",
                    "RESTORATION_METRICS.md",
                    "RESTORATION_METRICS.json",
                ):
                    aux_path = os.path.join(steps_dir, aux_name)
                    if os.path.exists(aux_path):
                        zf.write(aux_path, arcname=aux_name)

            st.download_button(
                "📦 Download all step images & INDEX.md (ZIP)",
                data=zip_buf.getvalue(),
                file_name="intermediate_steps.zip",
                mime="application/zip",
                use_container_width=True,
            )
