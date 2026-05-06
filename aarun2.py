import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("AETOS NDRE Engine V7 - Plant Background Removal + NDRE")

def load_gray(file):
    img = Image.open(file).convert("L")
    return np.array(img)

def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

def align_images_ecc(nir, red):
    try:
        if nir.shape != red.shape:
            red = cv2.resize(red, (nir.shape[1], nir.shape[0]))

        scale = 0.5
        nir_s = cv2.resize(nir, None, fx=scale, fy=scale)
        red_s = cv2.resize(red, None, fx=scale, fy=scale)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            1000,
            1e-6
        )

        cc, warp_matrix = cv2.findTransformECC(
            nir_s.astype(np.float32),
            red_s.astype(np.float32),
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5
        )

        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

        aligned_red = cv2.warpAffine(
            red.astype(np.float32),
            warp_matrix,
            (nir.shape[1], nir.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        st.success(f"ECC alignment successful. Correlation score: {cc:.4f}")
        return aligned_red

    except Exception as e:
        st.warning(f"ECC alignment failed. Continuing without alignment. Error: {e}")
        return red

def compute_ndre(nir, red):
    return (nir - red) / (nir + red + 1e-6)

def plant_mask_from_ndre(nir_norm, red_norm):
    ndre = compute_ndre(nir_norm, red_norm)

    nir8 = (nir_norm * 255).astype(np.uint8)

    # Remove sky / glare / dark shadows
    brightness_mask = (nir8 > 25) & (nir8 < 240)

    # Vegetation-like spectral relation
    # Plant should usually have NIR stronger than red edge
    spectral_mask = nir_norm > (red_norm * 0.85)

    # NDRE vegetation range
    ndre_mask = (ndre > -0.10) & (ndre < 0.75)

    mask = brightness_mask & spectral_mask & ndre_mask

    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove tiny noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean = np.zeros_like(mask)

    min_area = 300

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean > 0

def remove_background(gray_img, mask):
    plant_only = np.zeros_like(gray_img)
    plant_only[mask] = gray_img[mask]
    return plant_only

def create_ndre_colored(ndre, mask):
    ndre_display = np.full_like(ndre, np.nan)
    ndre_display[mask] = ndre[mask]
    return ndre_display

def create_overlay(base_gray, ndre, mask, alpha=0.55):
    base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    base_rgb = cv2.normalize(base_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ndre_clip = np.clip(ndre, -0.2, 0.6)
    ndre_norm = ((ndre_clip + 0.2) / 0.8 * 255).astype(np.uint8)

    color_map = cv2.applyColorMap(ndre_norm, cv2.COLORMAP_JET)

    overlay = base_rgb.copy()
    overlay[mask] = cv2.addWeighted(
        base_rgb[mask],
        1 - alpha,
        color_map[mask],
        alpha,
        0
    )

    overlay[~mask] = [0, 0, 0]

    return overlay

nir_file = st.file_uploader("Upload NIR Image")
red_file = st.file_uploader("Upload Red Edge Image")

if nir_file and red_file:

    nir_raw = load_gray(nir_file)
    red_raw = load_gray(red_file)

    st.subheader("Original Images")
    c1, c2 = st.columns(2)
    c1.image(nir_raw, caption="NIR Image", use_container_width=True)
    c2.image(red_raw, caption="Red Edge Image", use_container_width=True)

    nir_norm = normalize(nir_raw)
    red_norm = normalize(red_raw)

    red_aligned = align_images_ecc(nir_norm, red_norm)

    mask = plant_mask_from_ndre(nir_norm, red_aligned)

    st.subheader("Detected Plant Mask")
    st.image(
        (mask.astype(np.uint8) * 255),
        caption="White = Plant Pixels, Black = Background Removed",
        use_container_width=True
    )

    nir_plant_only = remove_background(nir_raw, mask)
    red_plant_only = remove_background(red_raw, mask)

    st.subheader("Background Removed Plant Images")
    c3, c4 = st.columns(2)
    c3.image(nir_plant_only, caption="NIR Plant Only", use_container_width=True)
    c4.image(red_plant_only, caption="Red Edge Plant Only", use_container_width=True)

    ndre = compute_ndre(nir_norm, red_aligned)

    valid_ndre = ndre[mask]

    if valid_ndre.size == 0:
        st.error("No plant pixels detected. Try swapping NIR and Red Edge images.")
        st.stop()

    valid_nir = nir_raw[mask]
    valid_red = red_raw[mask]

    mean_ndre = float(np.mean(valid_ndre))
    median_ndre = float(np.median(valid_ndre))
    min_ndre = float(np.min(valid_ndre))
    max_ndre = float(np.max(valid_ndre))

    mean_nir_dn = float(np.mean(valid_nir))
    mean_red_dn = float(np.mean(valid_red))

    plant_pixels = int(np.sum(mask))
    total_pixels = nir_raw.shape[0] * nir_raw.shape[1]
    plant_coverage = plant_pixels / total_pixels * 100

    stress_pixels = np.sum((ndre < 0.15) & mask)
    moderate_pixels = np.sum((ndre >= 0.15) & (ndre < 0.30) & mask)
    healthy_pixels = np.sum((ndre >= 0.30) & mask)

    stress_pct = stress_pixels / plant_pixels * 100
    moderate_pct = moderate_pixels / plant_pixels * 100
    healthy_pct = healthy_pixels / plant_pixels * 100

    st.subheader("NDRE Colored Plant Map with Scale")

    ndre_display = create_ndre_colored(ndre, mask)

    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.imshow(ndre_display, cmap="RdYlGn", vmin=-0.2, vmax=0.6)
    ax.axis("off")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDRE Scale")
    cbar.set_ticks([-0.2, 0.0, 0.15, 0.30, 0.45, 0.60])
    cbar.set_ticklabels([
        "Very Low\n-0.2",
        "Low\n0.0",
        "Stress\n0.15",
        "Moderate\n0.30",
        "Good\n0.45",
        "High\n0.60"
    ])

    st.pyplot(fig)

    st.subheader("NDRE Overlay - Plant Only")
    overlay = create_overlay(nir_raw, ndre, mask)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("NDRE Metrics")

    result = {
        "Mean NDRE": round(mean_ndre, 4),
        "Median NDRE": round(median_ndre, 4),
        "Min NDRE": round(min_ndre, 4),
        "Max NDRE": round(max_ndre, 4),
        "Mean NIR DN": round(mean_nir_dn, 2),
        "Mean Red Edge DN": round(mean_red_dn, 2),
        "Plant Pixels": plant_pixels,
        "Plant Coverage %": round(plant_coverage, 2),
        "Stress Pixels %": round(stress_pct, 2),
        "Moderate Pixels %": round(moderate_pct, 2),
        "Healthy Pixels %": round(healthy_pct, 2)
    }

    st.write(result)

    st.subheader("Interpretation")

    if mean_ndre < 0.15:
        st.error("Low NDRE: weak vegetation signal / stress / wrong image order / masking issue.")
    elif mean_ndre < 0.30:
        st.warning("Moderate NDRE: vegetation detected but chlorophyll signal is not very strong.")
    else:
        st.success("Good NDRE: strong vegetation/chlorophyll signal.")

    df = pd.DataFrame([result])
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "ndre_v7_results.csv",
        "text/csv"
    )
