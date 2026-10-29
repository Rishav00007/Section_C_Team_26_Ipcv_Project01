# Copy-Move Image Forgery Detection (CMFD)

A classical-computer-vision course project using SIFT feature descriptors, spatially constrained feature matching, Lowe's ratio test, RANSAC geometric verification, and approximate region localization.

> Educational baseline. Treat detections as suspicious regions, not proof of forgery. This method can miss forgeries and can flag naturally repeated patterns.

## 1. Setup (Mac / Windows / Linux)

Use Python 3.10–3.12 if possible for the smoothest package compatibility.

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run one image

```bash
python detect.py path/to/001_F.png --output outputs/demo
```

With a CoMoFoD binary mask (if the files are beside each other):

```bash
python detect.py path/to/001_F.png --mask path/to/001_B.png --output outputs/demo
```

Inspect `outputs/demo/overlay.png`, `predicted_mask.png`, `inlier_matches.jpg` (if matches exist), and `result.json`.

## 3. Dataset

Download the **small image category** from the official CoMoFoD download page. Read its release agreement and cite the original dataset paper. Extract it, then pass the extracted parent folder to the evaluation script. The script only uses base files named like `001_F.png`, `001_O.png`, `001_B.png`; postprocessed files such as `001_F_JC1.png` are ignored.

Official page: https://www.vcl.fer.hr/comofod/download.html
Dataset page / naming notes: https://www.vcl.fer.hr/comofod/comofod.html

## 4. Evaluate a subset

```bash
python evaluate_dataset.py /path/to/extracted/comofod --limit 20 --output outputs/eval_20

# Or choose the exact same IDs for each controlled parameter comparison:
python evaluate_dataset.py /path/to/extracted/comofod --ids 001,045,090,135,180 --output outputs/eval_fixed_ids
```

Each selected ID contributes its forged image (`F`) and original (`O`) when both are found. The script chooses evenly spaced IDs from the sorted available IDs, records them, creates per-image overlays/masks, and writes a per-image CSV plus summary JSON. Start with `--limit 5` to verify the paths and mask convention, then use 20–30 IDs if runtime allows.

## 5. Parameters to experiment with

Change one parameter at a time. Save outputs in different directories so experiments are not overwritten.

```bash
# Baseline
python detect.py sample_F.png --output outputs/exp_baseline

# More SIFT features
python detect.py sample_F.png --nfeatures 3000 --output outputs/exp_features3000

# Allow closer copies
python detect.py sample_F.png --min-distance 35 --output outputs/exp_dist35

# Stricter descriptor matching
python detect.py sample_F.png --ratio 0.70 --output outputs/exp_ratio070

# RANSAC tolerance
python detect.py sample_F.png --ransac-threshold 3 --output outputs/exp_ransac3

# More evidence required to flag
python detect.py sample_F.png --min-inliers 12 --output outputs/exp_inliers12

# Adjust the approximate mask expansion
python detect.py sample_F.png --dilation-size 25 --output outputs/exp_dilate25
```

Do not tune parameters on every evaluation image and report the best result without disclosure. Pick 3–5 development images, tune there, then evaluate the final chosen settings on a separate subset.

## 6. Output interpretation

- **Keypoints:** distinctive local image features found by SIFT.
- **Candidate matches:** distant features with similar descriptors that pass the ratio test.
- **RANSAC inliers:** matches that agree with one approximate translation/rotation/scale relationship.
- **Predicted mask:** convex hulls around the two inlier-point groups, dilated to approximate region extent. It is not a pixel-perfect segmentation.
- **Overlay:** predicted mask shown in red.
- **Image-level metrics:** forged/original classification based on the inlier threshold.
- **Pixel-level metrics:** compares predicted masks to the dataset's binary mask. Inspect several `B` masks first to establish whether the annotation covers the source region, copied region, or both; that affects interpretation.

## 7. Troubleshooting

- `Too few SIFT features`: try a more textured image or `--nfeatures 3000`.
- Too many false matches: try `--ratio 0.70`, increase `--min-distance`, or increase `--min-inliers`.
- Missed copies: try `--nfeatures 3000`, `--ratio 0.80`, or reduce `--min-inliers` cautiously.
- Mask looks too small: increase `--dilation-size`.
- Mask covers too much: decrease `--dilation-size` and inspect the match visualization.
- `No base image pairs found`: check that the extracted folder contains base names ending exactly in `_F`, `_O`, and `_B`; postprocessed files are deliberately ignored.

## Reference

Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013). *CoMoFoD — New Database for Copy-Move Forgery Detection*. Proceedings of the 55th International Symposium ELMAR, pp. 49–54.
