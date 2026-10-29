# Report outline — 4–5 pages

Replace every bracketed placeholder with your own measured values, outputs, and team details. Do not invent results.

## Title
Copy-Move Image Forgery Detection and Localization Using Classical Computer Vision

Team: [names and roll numbers] | Course: IPCV | Section C

## Abstract (about 120–160 words)
Digital images can be manipulated by copying a region and pasting it elsewhere in the same image, a process known as copy-move forgery. This project implements a classical computer-vision baseline for identifying suspected copy-move manipulation. The pipeline converts the image to grayscale, extracts SIFT keypoints and descriptors, matches descriptors between spatially separated regions of the same image, filters ambiguous matches using Lowe's ratio test, and applies RANSAC to retain matches consistent with an approximate similarity transformation. The inlier points are used to create an approximate suspicious-region mask and visual overlay. The method is evaluated on [N] image IDs from the CoMoFoD small-image dataset using image-level precision/recall/F1 and, where annotations are compatible, pixel-level precision/recall/F1/IoU. The final evaluation obtained [measured results]. The baseline is lightweight and interpretable, but repeated natural textures and low-texture manipulated regions can cause false positives or missed detections.

## 1. Introduction
Explain the real-world issue: images are used in news, social media, documentation, and evidence; a copied region may conceal an object or duplicate one. Define copy-move forgery as copying and pasting a region within the same image. Distinguish it briefly from splicing (pasting content from another image). State the objective: build a no-training, classical CV pipeline that finds geometrically consistent duplicated features and highlights approximate suspicious areas.

Problem statement: Given a single digital image, determine whether it contains a region duplicated elsewhere in the same image and localize the likely source/target areas. Because this is a heuristic baseline, outputs are treated as suspicion indicators rather than conclusive proof.

Objectives:
1. Extract local features using SIFT.
2. Match features while excluding nearby keypoints.
3. Filter matches and verify geometric consistency with RANSAC.
4. Visualize approximate duplicated regions and evaluate on labelled data.

## 2. Methodology
Describe the stages in order:
1. Input and grayscale conversion.
2. SIFT keypoint/descriptor extraction. A keypoint is a distinctive location; its descriptor summarizes local appearance.
3. Spatially constrained matching. The image is matched against itself, but same/nearby keypoints are disallowed so the matcher searches for similar content at a distant location.
4. Lowe's ratio test. A match is retained only if the best descriptor match is sufficiently better than its second-best alternative; the implementation uses ratio [actual value].
5. RANSAC with a partial-affine/similarity transform. It finds a group of matches consistent with an approximate translation, rotation, and uniform scale while rejecting outliers.
6. Localization. Convex hulls of inlier point groups are filled and dilated to create an approximate mask; a red overlay makes it interpretable.
7. Evaluation. Explain image-level labels and pixel-level metrics. State number of IDs/images, selected-ID policy, parameter settings, and how the binary mask annotations were checked.

Include a simple flow diagram made by the team: Input → Grayscale → SIFT → Spatially constrained matching → Ratio test → RANSAC → Region mask/overlay → Metrics.

## 3. Results and Discussion
Insert your own outputs, preferably:
- Figure 1: an input forged image and its CoMoFoD ground-truth mask.
- Figure 2: matched inlier features visualized side by side.
- Figure 3: input with predicted red overlay, alongside ground-truth mask.
- Table 1: image-level TP, FP, TN, FN, precision, recall, F1, accuracy.
- Table 2: pixel precision, pixel recall, pixel F1, IoU (only interpret if mask annotation semantics are compatible).
- Table 3: parameter experiment results.

Discuss: examples detected well; cases missed; false alarms on original images; effects of feature count, ratio threshold, minimum spatial distance, RANSAC threshold, minimum inliers, and mask dilation. Do not claim robust performance based on a tiny sample. State that a convex-hull mask is approximate, not exact segmentation.

## 4. Conclusion
Summarize what was implemented and measured. State strengths (no training, interpretable, handles some translation/rotation/scale transformations) and limitations (repeated patterns, smooth areas with few keypoints, transformations not represented well by a single similarity transform, approximate mask, possible false positives). Future work: block-based DCT/phase correlation, denser feature matching, better clustering/localization, or a learned model if permitted.

## References
1. Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013). “CoMoFoD — New Database for Copy-Move Forgery Detection.” Proc. 55th International Symposium ELMAR, pp. 49–54.
2. Lowe, D. G. (2004). “Distinctive Image Features from Scale-Invariant Keypoints.” International Journal of Computer Vision, 60, 91–110.
3. OpenCV documentation, “Feature Matching” and SIFT documentation: https://docs.opencv.org/
