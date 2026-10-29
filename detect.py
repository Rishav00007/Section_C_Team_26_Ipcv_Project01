#!/usr/bin/env python3
"""Classical copy-move forgery detection baseline using SIFT + RANSAC.

This is an educational baseline. It highlights suspected duplicated regions;
it cannot prove that an image is forged, and localization is approximate.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def make_spatial_match_mask(keypoints, min_distance: float) -> np.ndarray:
    """Allow descriptor matches only between keypoints far apart spatially.

    OpenCV's BFMatcher mask has one row per query descriptor and one column per
    train descriptor. A value of 0 forbids that pair; 255 allows it.
    """
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    n = len(points)
    allowed = np.full((n, n), 255, dtype=np.uint8)
    min_d2 = float(min_distance) ** 2
    for i, point in enumerate(points):
        delta = points - point
        d2 = np.einsum("ij,ij->i", delta, delta)
        allowed[i, d2 < min_d2] = 0
    return allowed


def convex_hull_or_points(points: np.ndarray, mask: np.ndarray, radius: int = 10) -> None:
    """Draw a convex hull for a point group, falling back to circles."""
    if len(points) >= 3:
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    else:
        for x, y in np.asarray(points).reshape(-1, 2):
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = pred > 0
    t = truth > 0
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"pixel_precision": precision, "pixel_recall": recall,
            "pixel_f1": f1, "pixel_iou": iou}


def detect_copy_move(
    image: np.ndarray,
    nfeatures: int = 2000,
    min_distance: float = 50.0,
    ratio: float = 0.75,
    ransac_threshold: float = 5.0,
    min_inliers: int = 8,
    dilation_size: int = 17,
) -> dict[str, Any]:
    """Detect suspicious duplicate regions in a BGR image.

    Pipeline: grayscale -> SIFT -> spatially constrained self-matching -> Lowe
    ratio test -> similarity-transform RANSAC -> approximate hull mask.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # SIFT generally works well on grayscale images. Light denoising is kept
    # minimal so we do not remove potentially useful local texture.
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    h, w = gray.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    if descriptors is None or len(keypoints) < 8:
        return {
            "is_suspected_forgery": False, "reason": "Too few SIFT features",
            "keypoint_count": len(keypoints), "candidate_match_count": 0,
            "inlier_count": 0, "mask": empty_mask,
            "inlier_matches": [], "keypoints": keypoints, "affine_matrix": None,
        }

    # Unlike ordinary image-to-image matching, we match an image against itself.
    # Forbid matches between nearby keypoints and self-matches, otherwise a
    # feature would simply match itself or an adjacent feature.
    spatial_mask = make_spatial_match_mask(keypoints, min_distance)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    try:
        knn_matches = matcher.knnMatch(descriptors, descriptors, k=2, mask=spatial_mask)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV could not apply the spatial matching mask. Install a recent "
            "opencv-contrib-python build: pip install --upgrade opencv-contrib-python"
        ) from exc

    candidates = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < ratio * n.distance:
            candidates.append(m)

    if len(candidates) < 3:
        return {
            "is_suspected_forgery": False, "reason": "Too few candidate matches",
            "keypoint_count": len(keypoints), "candidate_match_count": len(candidates),
            "inlier_count": 0, "mask": empty_mask,
            "inlier_matches": [], "keypoints": keypoints, "affine_matrix": None,
        }

    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    src = np.asarray([points[m.queryIdx] for m in candidates], dtype=np.float32)
    dst = np.asarray([points[m.trainIdx] for m in candidates], dtype=np.float32)

    # A partial affine transform models translation, rotation, and uniform scale.
    affine, inlier_flags = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=3000,
        confidence=0.99,
        refineIters=10,
    )

    if affine is None or inlier_flags is None:
        return {
            "is_suspected_forgery": False, "reason": "RANSAC found no consistent transform",
            "keypoint_count": len(keypoints), "candidate_match_count": len(candidates),
            "inlier_count": 0, "mask": empty_mask,
            "inlier_matches": [], "keypoints": keypoints, "affine_matrix": None,
        }

    flags = inlier_flags.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(candidates, flags) if keep]
    inlier_count = len(inlier_matches)
    suspicious = inlier_count >= min_inliers
    mask = empty_mask.copy()

    if suspicious:
        src_inliers = np.asarray([points[m.queryIdx] for m in inlier_matches], dtype=np.float32)
        dst_inliers = np.asarray([points[m.trainIdx] for m in inlier_matches], dtype=np.float32)
        convex_hull_or_points(src_inliers, mask)
        convex_hull_or_points(dst_inliers, mask)
        if dilation_size > 1:
            k = int(dilation_size)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

    return {
        "is_suspected_forgery": suspicious,
        "reason": "Geometrically consistent distant feature matches" if suspicious else "Inlier count below threshold",
        "keypoint_count": len(keypoints), "candidate_match_count": len(candidates),
        "inlier_count": inlier_count, "mask": mask,
        "inlier_matches": inlier_matches, "keypoints": keypoints,
        "affine_matrix": affine.tolist(),
    }



def make_overlay(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.42) -> np.ndarray:
    """Overlay suspicious regions in red, handling empty masks safely."""
    output = image.copy()
    active = mask > 0

    # Only blend when at least one pixel is flagged.
    if np.any(active):
        tint = np.zeros_like(image)
        tint[:, :, 2] = 255  # Red in OpenCV's BGR format.

        blended = cv2.addWeighted(
            image, 1.0 - alpha,
            tint, alpha,
            0
        )
        output[active] = blended[active]

    # Draw bounding boxes around sufficiently large suspicious regions.
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) >= 20:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                output, (x, y), (x + w, y + h),
                (0, 0, 255), 2
            )

    return output


def run_one(args: argparse.Namespace) -> int:
    input_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = detect_copy_move(
        image,
        nfeatures=args.nfeatures,
        min_distance=args.min_distance,
        ratio=args.ratio,
        ransac_threshold=args.ransac_threshold,
        min_inliers=args.min_inliers,
        dilation_size=args.dilation_size,
    )

    cv2.imwrite(str(output_dir / "predicted_mask.png"), result["mask"])
    cv2.imwrite(str(output_dir / "overlay.png"), make_overlay(image, result["mask"]))

    if result["inlier_matches"]:
        vis = cv2.drawMatches(
            image, result["keypoints"], image, result["keypoints"],
            result["inlier_matches"], None,
            matchColor=(0, 220, 0), singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(output_dir / "inlier_matches.jpg"), vis)

    metadata = {
        "image": str(input_path),
        "is_suspected_forgery": bool(result["is_suspected_forgery"]),
        "reason": result["reason"],
        "keypoint_count": int(result["keypoint_count"]),
        "candidate_match_count": int(result["candidate_match_count"]),
        "inlier_count": int(result["inlier_count"]),
        "parameters": {
            "nfeatures": args.nfeatures, "min_distance": args.min_distance,
            "ratio": args.ratio, "ransac_threshold": args.ransac_threshold,
            "min_inliers": args.min_inliers, "dilation_size": args.dilation_size,
        },
        "affine_matrix": result["affine_matrix"],
        "warning": "Educational baseline; output highlights suspected regions, not proof of forgery.",
    }
    if args.mask:
        gt = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Could not read ground-truth mask: {args.mask}")
        if gt.shape != result["mask"].shape:
            gt = cv2.resize(gt, (result["mask"].shape[1], result["mask"].shape[0]), interpolation=cv2.INTER_NEAREST)
        metadata["pixel_metrics_vs_provided_mask"] = compute_metrics(result["mask"], (gt > 127).astype(np.uint8) * 255)

    (output_dir / "result.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved output files to: {output_dir.resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIFT + RANSAC copy-move forgery detector")
    parser.add_argument("image", help="Path to one image")
    parser.add_argument("--output", default="outputs/demo", help="Output directory")
    parser.add_argument("--mask", default=None, help="Optional ground-truth binary mask for pixel metrics")
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--min-distance", type=float, default=50.0, help="Minimum spatial separation for matching keypoints")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test threshold")
    parser.add_argument("--ransac-threshold", type=float, default=5.0, help="RANSAC reprojection tolerance in pixels")
    parser.add_argument("--min-inliers", type=int, default=8, help="Minimum geometric inliers to flag suspicious content")
    parser.add_argument("--dilation-size", type=int, default=17, help="Odd-ish morphology kernel size for expanding the hull mask")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_one(parse_args()))
