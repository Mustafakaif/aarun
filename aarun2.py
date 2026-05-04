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
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(nir, None)
    kp2, des2 = orb.detectAndCompute(red, None)

    if des1 is None or des2 is None:
        return red

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    aligned = cv2.warpPerspective(red, H, (nir.shape[1], nir.shape[0]))
    return aligned

# Leaf mask
def mask_leaf(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold((blur*255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    return thresh == 0

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