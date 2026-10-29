#!/usr/bin/env python3
"""Evaluate the CMFD baseline on a small, reproducible CoMoFoD subset.

Looks recursively for base files named like 001_F.png, 001_O.png, 001_B.png.
Postprocessed variants such as 001_F_JC1.png are intentionally ignored.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from detect import detect_copy_move, make_overlay, compute_metrics

BASE_RE = re.compile(r"^(\d+)_([FOB])$", re.IGNORECASE)


def find_base_files(root: Path):
    found = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = BASE_RE.match(path.stem)
        if not match:
            continue
        ident, kind = match.group(1), match.group(2).upper()
        flag = cv2.IMREAD_GRAYSCALE if kind == "B" else cv2.IMREAD_COLOR
        if cv2.imread(str(path), flag) is None:
            print(f"Skipping unreadable base file: {path}")
            continue
        flag = cv2.IMREAD_GRAYSCALE if kind == "B" else cv2.IMREAD_COLOR
        if cv2.imread(str(path), flag) is None:
            print(f"Skipping unreadable base file: {path}")
            continue
        flag = cv2.IMREAD_GRAYSCALE if kind == "B" else cv2.IMREAD_COLOR
        if cv2.imread(str(path), flag) is None:
            print(f"Skipping unreadable base file: {path}")
            continue
        found.setdefault(ident, {})[kind] = path
    return found


def safe_read(path: Path, flag=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"OpenCV could not read file: {path}")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Folder containing extracted CoMoFoD files")
    parser.add_argument("--limit", type=int, default=20, help="Number of image IDs to sample; each ID contributes F and O if present")
    parser.add_argument("--ids", default=None, help="Optional comma-separated exact IDs, e.g. 001,045,120; overrides --limit")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--min-distance", type=float, default=50.0)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--dilation-size", type=int, default=17)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    all_files = find_base_files(root)
    eligible = [ident for ident, files in all_files.items() if "F" in files and "O" in files]
    eligible.sort(key=lambda value: int(value))
    if not eligible:
        raise RuntimeError(
            "No base image pairs found. Expected files like 001_F.png and 001_O.png. "
            "Check that you extracted the small-image dataset and passed its parent folder."
        )

    # Evenly-spaced IDs reduce the chance that a subset only samples one region
    # of an ID-sorted dataset. An explicit ID list makes parameter comparisons
    # reproducible and allows a separate small development set.
    if args.ids:
        wanted = [part.strip() for part in args.ids.split(",") if part.strip()]
        ids = []
        for raw in wanted:
            if not raw.isdigit():
                raise ValueError(f"Invalid ID {raw!r}; use numeric IDs such as 001,045")
            match_id = next((value for value in eligible if int(value) == int(raw)), None)
            if match_id is None:
                raise ValueError(f"ID {raw} was not found with both base F and O images")
            if match_id not in ids:
                ids.append(match_id)
    else:
        count = min(max(args.limit, 1), len(eligible))
        indices = np.linspace(0, len(eligible) - 1, count, dtype=int)
        ids = [eligible[i] for i in sorted(set(indices.tolist()))]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pixel_tp = pixel_fp = pixel_fn = 0
    image_tp = image_fp = image_tn = image_fn = 0

    for ident in ids:
        files = all_files[ident]
        for kind in ("F", "O"):
            if kind not in files:
                continue
            path = files[kind]
            image = safe_read(path, cv2.IMREAD_COLOR)
            result = detect_copy_move(
                image,
                nfeatures=args.nfeatures,
                min_distance=args.min_distance,
                ratio=args.ratio,
                ransac_threshold=args.ransac_threshold,
                min_inliers=args.min_inliers,
                dilation_size=args.dilation_size,
            )
            pred = result["mask"]
            predicted_positive = bool(result["is_suspected_forgery"])
            actual_positive = kind == "F"
            if actual_positive:
                image_tp += int(predicted_positive)
                image_fn += int(not predicted_positive)
            else:
                image_fp += int(predicted_positive)
                image_tn += int(not predicted_positive)

            gt_path = files.get("B") if actual_positive else None
            if actual_positive and gt_path is not None:
                gt = safe_read(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt.shape != pred.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                truth = (gt > 127).astype(np.uint8) * 255
                m = compute_metrics(pred, truth)
                pp, tt = pred > 0, truth > 0
                pixel_tp += int(np.logical_and(pp, tt).sum())
                pixel_fp += int(np.logical_and(pp, ~tt).sum())
                pixel_fn += int(np.logical_and(~pp, tt).sum())
                gt_available = True
            else:
                truth = np.zeros_like(pred)
                if not actual_positive:  # original image is an authentic negative
                    pp = pred > 0
                    pixel_fp += int(pp.sum())
                m = {"pixel_precision": "", "pixel_recall": "", "pixel_f1": "", "pixel_iou": ""}
                gt_available = False

            prefix = f"{ident}_{kind}"
            cv2.imwrite(str(out_dir / f"{prefix}_predicted_mask.png"), pred)
            cv2.imwrite(str(out_dir / f"{prefix}_overlay.png"), make_overlay(image, pred))
            rows.append({
                "image_id": ident, "kind": kind, "filename": path.name,
                "actual_forgery": int(actual_positive), "predicted_forgery": int(predicted_positive),
                "keypoints": result["keypoint_count"], "candidate_matches": result["candidate_match_count"],
                "ransac_inliers": result["inlier_count"], "ground_truth_mask_found": int(gt_available),
                **m,
            })
            print(f"{prefix}: actual={'forged' if actual_positive else 'original'}, "
                  f"predicted={'suspected' if predicted_positive else 'not flagged'}, "
                  f"keypoints={result['keypoint_count']}, candidates={result['candidate_match_count']}, "
                  f"inliers={result['inlier_count']}")

    csv_path = out_dir / "per_image_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    p = pixel_tp / (pixel_tp + pixel_fp) if pixel_tp + pixel_fp else 0.0
    r = pixel_tp / (pixel_tp + pixel_fn) if pixel_tp + pixel_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = pixel_tp / (pixel_tp + pixel_fp + pixel_fn) if pixel_tp + pixel_fp + pixel_fn else 1.0
    total_cls = image_tp + image_fp + image_tn + image_fn
    summary = {
        "dataset_root": str(root), "selected_ids": ids,
        "images_evaluated": len(rows), "forged_images_detected": image_tp,
        "forged_images_missed": image_fn, "original_images_false_alarm": image_fp,
        "original_images_correctly_not_flagged": image_tn,
        "image_level_accuracy": (image_tp + image_tn) / total_cls if total_cls else 0.0,
        "image_level_precision": image_tp / (image_tp + image_fp) if image_tp + image_fp else 0.0,
        "image_level_recall": image_tp / (image_tp + image_fn) if image_tp + image_fn else 0.0,
        "image_level_f1": (2 * image_tp / (2 * image_tp + image_fp + image_fn)) if (2 * image_tp + image_fp + image_fn) else 0.0,
        "aggregate_pixel_precision": p, "aggregate_pixel_recall": r,
        "aggregate_pixel_f1": f1, "aggregate_pixel_iou": iou,
        "pixel_metrics_note": "Pixel metrics aggregate forged-image masks plus false-positive pixels on originals. Verify the meaning of CoMoFoD B masks visually before interpreting scores.",
        "parameters": {"nfeatures": args.nfeatures, "min_distance": args.min_distance,
                       "ratio": args.ratio, "ransac_threshold": args.ransac_threshold,
                       "min_inliers": args.min_inliers, "dilation_size": args.dilation_size},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {csv_path.resolve()} and {(out_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
