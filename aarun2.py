import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide")
st.title("AETOS NDRE Engine V3")

# Load image
def load_image(file):
    img = Image.open(file).convert("L")
    return np.array(img)

# Normalize DN
def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

# Align images
def align_images(nir, red):
    try:
        # Convert normalized float images to float32
        nir_f = nir.astype(np.float32)
        red_f = red.astype(np.float32)

        # ECC needs same size
        if nir_f.shape != red_f.shape:
            red_f = cv2.resize(red_f, (nir_f.shape[1], nir_f.shape[0]))

        # Motion model: translation + rotation + scale
        warp_mode = cv2.MOTION_EUCLIDEAN

        # Initial warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # ECC stopping criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            5000,
            1e-7
        )

        # Find alignment
        cc, warp_matrix = cv2.findTransformECC(
            nir_f,
            red_f,
            warp_matrix,
            warp_mode,
            criteria,
            None,
            5
        )

        # Apply alignment
        aligned_red = cv2.warpAffine(
            red_f,
            warp_matrix,
            (nir_f.shape[1], nir_f.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        st.success(f"ECC alignment successful. Correlation score: {cc:.4f}")
        return aligned_red

    except Exception as e:
        st.warning(f"ECC alignment failed. Continuing without alignment. Error: {e}")
        return red
        # Leaf mask
def mask_leaf(img):
    img8 = (img * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(img8, (7, 7), 0)

    # Leaf is darker than background in your images
    _, mask = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only largest object = leaf
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask > 0

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = labels == largest_label

    return clean_mask
# NDRE calculation
def compute_ndre(nir, red):
    return (nir - red) / (nir + red + 1e-6)

# Upload
nir_file = st.file_uploader("Upload NIR Image")
red_file = st.file_uploader("Upload Red Edge Image")

if nir_file and red_file:

    nir = load_image(nir_file)
    red = load_image(red_file)

    st.subheader("Original Images")
    col1, col2 = st.columns(2)
    col1.image(nir, caption="NIR", use_container_width=True)
    col2.image(red, caption="Red Edge", use_container_width=True)

    # Normalize
    nir_norm = normalize(nir)
    red_norm = normalize(red)

    # Align
    red_aligned = align_images(nir_norm, red_norm)

    st.subheader("Aligned Images")
    col3, col4 = st.columns(2)
    col3.image(nir_norm, caption="Normalized NIR", use_container_width=True)
    col4.image(red_aligned, caption="Aligned Red Edge", use_container_width=True)

    # Mask
    mask = mask_leaf(nir_norm)

    # NDRE
    ndre = compute_ndre(nir_norm, red_aligned)
    ndre_masked = ndre[mask]

    # Metrics
    mean_ndre = float(np.mean(ndre_masked))
    min_ndre = float(np.min(ndre_masked))
    max_ndre = float(np.max(ndre_masked))

    st.subheader("NDRE Heatmap")
    st.image(ndre, clamp=True)

    st.subheader("NDRE Metrics")
    st.write({
        "Mean NDRE": mean_ndre,
        "Min NDRE": min_ndre,
        "Max NDRE": max_ndre,
        "Leaf Pixels": int(np.sum(mask))
    })

    df = pd.DataFrame([{
        "Mean NDRE": mean_ndre,
        "Min NDRE": min_ndre,
        "Max NDRE": max_ndre,
        "Leaf Pixels": int(np.sum(mask))
    }])

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "ndre_results.csv")
