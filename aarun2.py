import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("AETOS NDRE Image Generator V4")

def load_image(file):
    img = Image.open(file).convert("L")
    return np.array(img)

def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

def align_images_ecc(nir, red):
    try:
        if nir.shape != red.shape:
            red = cv2.resize(red, (nir.shape[1], nir.shape[0]))

        # resize for faster ECC
        scale = 0.5
        nir_small = cv2.resize(nir, None, fx=scale, fy=scale)
        red_small = cv2.resize(red, None, fx=scale, fy=scale)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            1000,
            1e-6
        )

        cc, warp_matrix = cv2.findTransformECC(
            nir_small.astype(np.float32),
            red_small.astype(np.float32),
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5
        )

        # scale warp back to original size
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

def mask_leaf(nir_norm):
    img8 = (nir_norm * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(img8, (7, 7), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask > 0

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = labels == largest_label

    return clean_mask

def compute_ndre(nir, red):
    return (nir - red) / (nir + red + 1e-6)

nir_file = st.file_uploader("Upload NIR Image")
red_file = st.file_uploader("Upload Red Edge Image")

if nir_file and red_file:

    nir = load_image(nir_file)
    red = load_image(red_file)

    st.subheader("Original Images")
    c1, c2 = st.columns(2)
    c1.image(nir, caption="NIR Image", use_container_width=True)
    c2.image(red, caption="Red Edge Image", use_container_width=True)

    nir_norm = normalize(nir)
    red_norm = normalize(red)

    red_aligned = align_images_ecc(nir_norm, red_norm)

    mask = mask_leaf(nir_norm)

    st.subheader("Detected Leaf Mask")
    st.image((mask.astype(np.uint8) * 255), caption="White = Leaf Area", use_container_width=True)

    ndre = compute_ndre(nir_norm, red_aligned)

    ndre_leaf_only = np.full_like(ndre, np.nan)
    ndre_leaf_only[mask] = ndre[mask]

    valid_ndre = ndre[mask]
    valid_nir = nir[mask]
    valid_red = red[mask]

    mean_ndre = float(np.nanmean(valid_ndre))
    median_ndre = float(np.nanmedian(valid_ndre))
    min_ndre = float(np.nanmin(valid_ndre))
    max_ndre = float(np.nanmax(valid_ndre))

    mean_nir_dn = float(np.mean(valid_nir))
    mean_red_dn = float(np.mean(valid_red))

    st.subheader("Cropler-like NDRE Colored Image")

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.RdYlGn
    cax = ax.imshow(ndre_leaf_only, cmap=cmap, vmin=-0.2, vmax=0.6)
    ax.axis("off")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="NDRE")
    st.pyplot(fig)

    st.subheader("NDRE Metrics")

    result = {
        "Mean NDRE": round(mean_ndre, 4),
        "Median NDRE": round(median_ndre, 4),
        "Min NDRE": round(min_ndre, 4),
        "Max NDRE": round(max_ndre, 4),
        "Mean NIR DN": round(mean_nir_dn, 2),
        "Mean Red Edge DN": round(mean_red_dn, 2),
        "Leaf Pixels": int(np.sum(mask))
    }

    st.write(result)

    df = pd.DataFrame([result])
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "ndre_results.csv",
        "text/csv"
    )
