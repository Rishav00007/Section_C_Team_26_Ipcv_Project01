#!/usr/bin/env python3
"""Streamlit frontend for the SIFT + RANSAC copy-move forgery detector.

Place this file in the same folder as detect.py, then run:
    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import streamlit as st

from detect import compute_metrics, detect_copy_move, make_overlay

# Optional: real clipboard paste (pip install streamlit-paste-button)
try:
    from streamlit_paste_button import paste_image_button
    HAS_PASTE = True
except ImportError:
    HAS_PASTE = False

st.set_page_config(
    page_title="Copy-move forgery detector",
    page_icon="🔍",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
  --ink: #14213d;
  --muted: #5b6577;
  --paper: #f3f5f8;
  --panel: #ffffff;
  --line: #d7dce5;
  --teal: #0f766e;
  --alert: #b42318;
  --alert-bg: #fdecea;
  --ok: #1a7f4b;
  --ok-bg: #e7f5ed;
}

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--ink);
}
.stApp { background: var(--paper); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

h1.page-title {
  font-family: 'IBM Plex Serif', Georgia, serif;
  font-weight: 600;
  font-size: 2.3rem;
  letter-spacing: -0.01em;
  margin: 0 0 .25rem 0;
}
p.page-sub { color: var(--muted); font-size: 1.02rem; margin: 0 0 1.6rem 0; max-width: 62ch; }

.panel-title {
  font-weight: 600; font-size: 1.05rem; margin: 0 0 .5rem 0;
  display: flex; align-items: center; gap: .5rem;
}
.panel-title .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--teal); display: inline-block; }
.panel-title .dot.red { background: var(--alert); }

.verdict {
  border-radius: 10px; padding: 1rem 1.2rem; margin: .4rem 0 1.2rem 0;
  border-left: 6px solid; display: flex; flex-direction: column; gap: .15rem;
}
.verdict.bad { background: var(--alert-bg); border-color: var(--alert); }
.verdict.good { background: var(--ok-bg); border-color: var(--ok); }
.verdict .v-head { font-family: 'IBM Plex Serif', serif; font-weight: 600; font-size: 1.35rem; }
.verdict.bad .v-head { color: var(--alert); }
.verdict.good .v-head { color: var(--ok); }
.verdict .v-body { color: var(--muted); font-size: .95rem; }

.placeholder {
  border: 2px dashed var(--line); border-radius: 10px; background: #fafbfc;
  min-height: 320px; display: flex; align-items: center; justify-content: center;
  text-align: center; color: var(--muted); padding: 2rem;
}

[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--line); }
[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: .8rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--muted); }
section[data-testid="stSidebar"] { background: #e9edf3; }

.stButton > button[kind="primary"] {
  background: var(--teal); border: none; color: #fff; font-weight: 600;
  border-radius: 8px; padding: .65rem 1.4rem;
}
.stButton > button[kind="primary"]:hover { background: #0b5f59; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 3px solid #7cc4bd; outline-offset: 2px;
}
.fineprint { color: var(--muted); font-size: .85rem; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else b""


def decode_upload(file) -> np.ndarray | None:
    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def pil_to_bgr(pil_img) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def transform_summary(M) -> dict | None:
    if M is None:
        return None
    M = np.asarray(M)
    return {
        "scale": round(float(np.hypot(M[0, 0], M[1, 0])), 3),
        "rotation_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
        "shift_x_px": round(float(M[0, 2]), 1),
        "shift_y_px": round(float(M[1, 2]), 1),
    }


# --------------------------------------------------------------------------
# Sidebar: detector settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Detector settings")
    st.caption("Defaults match detect.py. Change one value at a time to see its effect.")

    nfeatures = st.slider("Max keypoints", 500, 8000, 2000, 250,
                          help="More keypoints find clones in smoother areas but run slower.")
    min_distance = st.slider("Minimum distance between matches (px)", 10, 200, 50, 5,
                             help="Ignores matches between keypoints that sit close together.")
    ratio = st.slider("Match strictness (ratio test)", 0.40, 0.95, 0.75, 0.01,
                      help="Lower values keep fewer, cleaner matches.")
    ransac_threshold = st.slider("Position tolerance (px)", 1.0, 12.0, 5.0, 0.5,
                                 help="How far a match may sit from the fitted transform and still count.")
    min_inliers = st.slider("Matches needed to flag an image", 4, 40, 8, 1,
                            help="Higher values reduce false alarms but can miss small clones.")
    dilation_size = st.slider("Region padding (px)", 1, 61, 17, 2,
                              help="Expands the highlighted region around matched points.")

    st.divider()
    st.subheader("Display")
    alpha = st.slider("Highlight opacity", 0.1, 0.9, 0.42, 0.02)
    max_side = st.select_slider("Resize large images to", options=[0, 800, 1024, 1600, 2400],
                                value=1600,
                                format_func=lambda v: "Original size" if v == 0 else f"{v} px longest side")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<h1 class="page-title">Copy-move forgery detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Add a photo and the detector looks for parts of the image that were '
    "copied and pasted elsewhere in the same image. Suspected regions are highlighted in red.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
image: np.ndarray | None = None
source_name = ""

options = ["Upload a file", "Paste from clipboard"] if HAS_PASTE else ["Upload a file"]
mode = st.radio("Image source", options, horizontal=True, label_visibility="collapsed")

if mode == "Upload a file":
    up = st.file_uploader(
        "Drop an image here, or browse (you can also press Ctrl+V on the uploader to paste)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    )
    if up is not None:
        image = decode_upload(up)
        source_name = up.name
        if image is None:
            st.error("That file could not be read as an image. Try a PNG or JPEG.")
else:
    paste = paste_image_button("Paste image from clipboard", background_color="#0f766e",
                               hover_background_color="#0b5f59")
    if paste.image_data is not None:
        image = pil_to_bgr(paste.image_data)
        source_name = "pasted image"
    else:
        st.info("Copy an image (for example with Win+Shift+S or Cmd+Ctrl+Shift+4), then click the button.")

if not HAS_PASTE:
    st.caption("For a one-click clipboard button, run: pip install streamlit-paste-button")

if image is not None:
    image = limit_size(image, max_side)
    image_id = hashlib.md5(image.tobytes() + str(image.shape).encode()).hexdigest()
else:
    image_id = None

run = st.button("Detect forgery", type="primary", disabled=image is None)

if run and image is not None:
    with st.spinner("Matching features and checking for duplicated regions..."):
        try:
            res = detect_copy_move(
                image,
                nfeatures=nfeatures,
                min_distance=float(min_distance),
                ratio=float(ratio),
                ransac_threshold=float(ransac_threshold),
                min_inliers=int(min_inliers),
                dilation_size=int(dilation_size),
            )
            st.session_state["result"] = {"id": image_id, "res": res}
        except Exception as exc:  # surface OpenCV or input errors clearly
            st.session_state.pop("result", None)
            st.error(f"Detection failed: {exc}")

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
stored = st.session_state.get("result")
has_result = image is not None and stored is not None and stored["id"] == image_id
res = stored["res"] if has_result else None

if has_result:
    if res["is_suspected_forgery"]:
        st.markdown(
            f'<div class="verdict bad"><div class="v-head">Possible copy-move forgery</div>'
            f'<div class="v-body">{res["inlier_count"]} feature matches agree on one copy transform. '
            f"Red areas show the suspected source and pasted regions.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict good"><div class="v-head">No duplicated region flagged</div>'
            f'<div class="v-body">{res["reason"]}. This does not prove the image is authentic; '
            f"small or smooth-area clones can be missed.</div></div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Keypoints found", res["keypoint_count"])
    c2.metric("Candidate matches", res["candidate_match_count"])
    c3.metric("Matches agreeing on one copy", res["inlier_count"])

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="panel-title"><span class="dot"></span>Input image</div>',
                unsafe_allow_html=True)
    if image is not None:
        st.image(to_rgb(image), use_container_width=True,
                 caption=f"{source_name} · {image.shape[1]}×{image.shape[0]} px")
    else:
        st.markdown('<div class="placeholder">Your image will appear here.</div>',
                    unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-title"><span class="dot red"></span>Detected regions</div>',
                unsafe_allow_html=True)
    if has_result:
        overlay = make_overlay(image, res["mask"], alpha=alpha)
        st.image(to_rgb(overlay), use_container_width=True,
                 caption="Red tint and boxes mark suspected duplicated areas")
        st.download_button("Download result image", to_png_bytes(overlay),
                           file_name="forgery_overlay.png", mime="image/png")
    else:
        msg = "Press Detect forgery to see the result." if image is not None \
            else "The highlighted result will appear here."
        st.markdown(f'<div class="placeholder">{msg}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Extra views
# --------------------------------------------------------------------------
if has_result:
    st.write("")
    tab_mask, tab_matches, tab_details, tab_eval = st.tabs(
        ["Predicted mask", "Matched features", "Details", "Score against a mask"]
    )

    with tab_mask:
        st.image(res["mask"], use_container_width=True, clamp=True,
                 caption="White = pixels flagged as part of a duplicated region")
        st.download_button("Download mask", to_png_bytes(res["mask"]),
                           file_name="predicted_mask.png", mime="image/png")

    with tab_matches:
        if res["inlier_matches"]:
            vis = cv2.drawMatches(
                image, res["keypoints"], image, res["keypoints"],
                res["inlier_matches"], None,
                matchColor=(0, 200, 0), singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            st.image(to_rgb(vis), use_container_width=True,
                     caption="Green lines connect points that look identical but sit in different places")
        else:
            st.info("No consistent matches to draw for this image.")

    with tab_details:
        tf = transform_summary(res["affine_matrix"])
        if tf:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Scale", tf["scale"])
            d2.metric("Rotation", f'{tf["rotation_deg"]}°')
            d3.metric("Shift x", f'{tf["shift_x_px"]} px')
            d4.metric("Shift y", f'{tf["shift_y_px"]} px')
            st.caption("A scale near 1.0 and rotation near 0° means the region was copied without resizing or turning.")
        st.json({
            "is_suspected_forgery": bool(res["is_suspected_forgery"]),
            "reason": res["reason"],
            "keypoint_count": int(res["keypoint_count"]),
            "candidate_match_count": int(res["candidate_match_count"]),
            "inlier_count": int(res["inlier_count"]),
            "affine_matrix": res["affine_matrix"],
        })
        st.download_button(
            "Download result as JSON",
            json.dumps({
                "image": source_name,
                "is_suspected_forgery": bool(res["is_suspected_forgery"]),
                "reason": res["reason"],
                "keypoint_count": int(res["keypoint_count"]),
                "candidate_match_count": int(res["candidate_match_count"]),
                "inlier_count": int(res["inlier_count"]),
                "affine_matrix": res["affine_matrix"],
            }, indent=2),
            file_name="result.json", mime="application/json",
        )

    with tab_eval:
        st.write("Upload the ground-truth mask (for CoMoFoD, the `_B` file) to measure how well the red region matches it.")
        gt_file = st.file_uploader("Ground-truth mask", type=["png", "jpg", "jpeg", "bmp"], key="gt")
        if gt_file is not None:
            gt = cv2.imdecode(np.frombuffer(gt_file.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                st.error("Could not read that mask.")
            else:
                pred = res["mask"]
                if gt.shape != pred.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                truth = (gt > 127).astype(np.uint8) * 255
                m = compute_metrics(pred, truth)
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Precision", f'{m["pixel_precision"]:.3f}')
                e2.metric("Recall", f'{m["pixel_recall"]:.3f}')
                e3.metric("F1", f'{m["pixel_f1"]:.3f}')
                e4.metric("IoU", f'{m["pixel_iou"]:.3f}')
                st.image(truth, caption="Ground-truth mask", use_container_width=True, clamp=True)

st.markdown(
    '<p class="fineprint">This is a classical baseline (SIFT features with RANSAC). '
    "It highlights suspected duplicated regions and cannot prove that an image was edited.</p>",
    unsafe_allow_html=True,
)