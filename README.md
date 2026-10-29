# Copy-Move Image Forgery Detection (CMFD)

**Course:** Image Processing and Computer Vision (IPCV)\
**Section:** C\
**Project type:** Image processing and computer vision course project\
**Primary language:** Python\
**Core library:** OpenCV


## Table of contents

-   [Team and contributions](#team-and-contributions)
-   [Problem statement](#problem-statement)
-   [Project objectives](#project-objectives)
-   [Scope](#scope)
-   [How the method works](#how-the-method-works)
-   [Technology stack](#technology-stack)
-   [Repository structure](#repository-structure)
-   [Dataset](#dataset)
-   [Requirements](#requirements)
-   [Step-by-step: how to run the code](#step-by-step-how-to-run-the-code)
-   [Run locally on Windows](#run-locally-on-windows)
-   [Run in Google Colab](#run-in-google-colab)
-   [Command-line usage](#command-line-usage)
-   [Evaluation and experiments](#evaluation-and-experiments)
-   [Output files](#output-files)
-   [How to improve and compare
    results](#how-to-improve-and-compare-results)
-   [Results](#results)
-   [Limitations](#limitations)
-   [Reproducibility checklist](#reproducibility-checklist)
-   [References and dataset
    acknowledgement](#references-and-dataset-acknowledgement)
-   [License and dataset usage](#license-and-dataset-usage)

------------------------------------------------------------------------

## Team and contributions

**Team number:** \[26\]\
**Institution / department:** \[CSE\]\
**Submission date:** 30 October 2026

  -----------------------------------------------------------------------
  Team member             Student ID              Contribution
  ----------------------- ----------------------- -----------------------
  \[Rishav Ranjan\] \[20233345\]            \[Dataset preparation, preprocessing, Frontend Design and Integration\]

  \[Manu Singhal\] \[20233533\]            \[SIFT feature matching and RANSAC\]

  \[Rishabh Jain\] \[20233232\]            \[Evaluation, experiments, and visualizations\]

  \[Nakul Tiwari\]  \[20233194\]            \[Testing, report, and presentation\]
  -----------------------------------------------------------------------


## Problem statement

Copy-move image forgery is a form of image manipulation in which a
region of an image is copied and pasted into another location within the
**same image**. It may be used to duplicate an object, hide an object,
or make a scene appear different from the original. Because the copied
pixels originate from the same image, their colour, noise, and texture
can be consistent with the surrounding content, making this manipulation
difficult to identify by visual inspection alone.

**Problem statement:** Design and implement a computer-vision system
that analyses an input image for duplicated regions that may indicate
copy-move forgery, identifies geometrically consistent feature matches,
and produces a visual estimate of the suspicious region. Evaluate the
system on labelled examples and discuss its successes and failure cases.

This is a decision-support prototype, not a forensic certification tool.
A detected region is evidence of a possible duplicated pattern, not
proof that an image has been forged.

## Project objectives

1.  Implement a classical, explainable copy-move detection pipeline
    using Python and OpenCV.
2.  Detect distinctive local image features and compare their
    descriptors within the same image.
3.  Filter ambiguous matches and use geometric verification to reduce
    inconsistent matches.
4.  Visualise suspicious regions with feature-match images, masks, and
    overlays.
5.  Evaluate image-level detection and pixel-level localisation against
    dataset labels where available.
6.  Experiment with parameters and discuss the effect on false
    positives, missed detections, and localisation quality.

## Scope

This project focuses on **copy-move forgery**: copying a region from one
part of an image and placing it elsewhere in that same image. It is not
designed to reliably detect every manipulation type, such as splicing
content from another photograph, AI-generated edits, or all forms of
retouching.

## How the method works

The implementation uses a classical feature-based approach built around
SIFT (Scale-Invariant Feature Transform), descriptor matching, and
RANSAC geometric verification.

1.  **Input and preprocessing:** Load the image and convert it to
    grayscale for feature extraction.
2.  **Feature extraction:** SIFT identifies local keypoints and computes
    descriptors that represent the appearance around each keypoint.
3.  **Descriptor matching:** A nearest-neighbour matcher finds similar
    descriptors in the same image. Spatially close or otherwise
    unsuitable self-matches are filtered according to the
    implementation.
4.  **Lowe's ratio test:** Compare the best and second-best descriptor
    matches. A stricter ratio threshold rejects more ambiguous matches,
    but may also remove valid matches.
5.  **Geometric verification with RANSAC:** Estimate a geometric
    relationship between matched point sets and retain matches
    consistent with the estimated transformation. This helps reject
    unrelated matches.
6.  **Suspicious-region localisation:** Use the verified matches to
    estimate a region, such as a convex hull, and expand/smooth the
    estimated region using morphological dilation. This creates an
    approximate predicted mask.
7.  **Visualisation and decision:** Save the predicted mask, an overlay
    on the input image, matching visualisations when available, and
    result metadata. The suspicion decision depends on the
    implementation's match/inlier criteria.

The resulting mask is an approximation. Feature points do not
necessarily cover the full copied object, and dilation can extend the
predicted region beyond the manipulated pixels.

### Pipeline

``` text
Input image
    |
    v
Grayscale preprocessing
    |
    v
SIFT keypoints and descriptors
    |
    v
Descriptor matching within the image
    |
    v
Spatial filtering + Lowe's ratio test
    |
    v
RANSAC geometric verification
    |
    v
Suspicious-region mask and overlay
    |
    v
Evaluation against labelled masks (when available)
```

## Technology stack

-   **Python 3:** implementation and experiment orchestration.
-   **OpenCV (`cv2`):** image processing, SIFT, descriptor matching,
    geometric verification, masks, and visualisations.
-   **NumPy:** array operations and mask calculations.
-   **Matplotlib:** viewing and creating figures in the notebook
    workflow.
-   **Pandas:** inspecting and exporting evaluation tables.
-   **Google Colab (optional):** hosted Python environment that avoids
    local package setup.
-   **Streamlit (`app.py`):** interactive browser interface for manual
    inspection and quick visual testing.

The detector does not require a GPU.

## Repository structure

The exact folders can differ depending on where you extract the dataset
and store outputs. A typical layout is:

``` text
CMFD-project/
├── app.py
├── detect.py
├── evaluate_dataset.py
├── requirements.txt
├── README.md
├── report_outline.md
├── DATASET.md
├── comofod_data/
│   └── ... extracted CoMoFoD files
└── outputs/
    ├── demo_001/
    │   ├── overlay.png
    │   ├── predicted_mask.png
    │   ├── inlier_matches.jpg
    │   └── result.json
    ├── per_image_results.csv
    └── summary.json
```

-   `detect.py`: runs the core SIFT + RANSAC detection pipeline on a
    single image.
-   `app.py`: interactive Streamlit frontend for uploading or pasting an
    image and inspecting the suspicious region visually.
-   `evaluate_dataset.py`: batch evaluator for running multiple images and
    summarising dataset-level results.
-   `outputs/`: generated result images and evaluation files.



## Dataset

The recommended dataset is **CoMoFoD (Copy-Move Forgery Detection)**, a
research dataset designed for copy-move forgery detection. The project
workflow initially uses the base original, forged, and binary-mask
files:

-   `*_O.*`: original image.
-   `*_F.*`: forged image.
-   `*_B.*`: binary ground-truth mask, when supplied for that image set.

For example, a matching set might include `001_O.png`, `001_F.png`, and
`001_B.png`. The actual extension and directory structure may differ;
confirm them after extraction.

### Obtain the dataset

1.  Visit the [official CoMoFoD download
    page](https://www.vcl.fer.hr/comofod/download.html).
2.  Read the [release
    agreement](https://www.vcl.fer.hr/comofod/release_agreemen.html)
    before downloading.
3.  Download the small-image category if following the supplied starter
    workflow.
4.  Extract it locally or in the Colab runtime.
5.  Confirm that original, forged, and mask files can be matched by ID
    before running the evaluator.



## Requirements

-   Python 3.9 or newer is a reasonable starting point; use a Python
    version compatible with the installed requirements.
-   `pip`
-   The packages listed in `requirements.txt`
-   The project source files (`detect.py`, `evaluate_dataset.py`, and
    `app.py`)
-   Dataset files, downloaded separately

## Step-by-step: how to run the code

There are two main ways to use the project:

1. Command-line mode for a single image or dataset evaluation.
2. GUI mode using `app.py` for interactive testing.

### 1. Clone or open the project folder

Open a terminal in the project root directory.

``` powershell
cd "path\to\CMFD-project"
```

Make sure the following files are present:

-   `detect.py`
-   `evaluate_dataset.py`
-   `app.py`
-   `requirements.txt`

### 2. Create a Python virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use the virtual environment Python
 directly or activate from a shell that allows it.

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If needed, install the core packages manually:

``` powershell
python -m pip install opencv-python numpy matplotlib pandas streamlit
```

### 4. Check that the detector works

Verify OpenCV and SIFT are available:

``` powershell
python -c "import cv2; print('OpenCV:', cv2.__version__); print('SIFT available:', hasattr(cv2, 'SIFT_create'))"
```

This should print `True` for SIFT support.

### 5. Run the detector on a single image from the command line

From the project root:

``` powershell
python detect.py "comofod_data/001_F.png" --output "outputs/demo_001"
```

If you also have a ground-truth mask and want pixel-level comparison:

``` powershell
python detect.py "comofod_data/001_F.png" --mask "comofod_data/001_B.png" --output "outputs/demo_001"
```

This command:

- loads the input image
- runs the SIFT + RANSAC copy-move detection pipeline
- saves:
  - `predicted_mask.png`
  - `overlay.png`
  - `inlier_matches.jpg` (when matches are available)
  - `result.json`

### 6. Run the dataset evaluator

If your CoMoFoD dataset is inside `comofod_data`, evaluate several images:

``` powershell
python evaluate_dataset.py "comofod_data" --output "outputs/evaluation"
```

You can also preview supported options:

``` powershell
python evaluate_dataset.py --help
```

This script scans the dataset, evaluates images, and writes:

-   `per_image_results.csv`
-   `summary.json`

### 7. Run the interactive Streamlit app (`app.py`)

The app is the easiest way to test the detector visually without writing
command-line arguments each time.

``` powershell
streamlit run app.py
```

Then in the browser:

1.  Upload an image or paste one from the clipboard.
2.  Adjust the detector parameters in the left sidebar.
3.  Click **Detect forgery**.
4.  Inspect:
    - input image
    - suspicious region overlay
    - predicted mask
    - matched feature view
    - optional score against a ground-truth mask

#### Role of `app.py`

`app.py` is the interactive frontend for the project. It does not contain
the main detection algorithm itself; it calls functions from `detect.py`
such as:

-   `detect_copy_move(...)`
-   `make_overlay(...)`
-   `compute_metrics(...)`

This makes it useful for:

- quick manual testing
- parameter tuning by eye
- comparing outputs on a single sample image
- showing the result in a user-friendly interface

### 8. Typical workflow

For most users, the recommended sequence is:

``` powershell
python detect.py "comofod_data/001_F.png" --mask "comofod_data/001_B.png" --output "outputs/demo_001"
streamlit run app.py
```

Use:

-   `detect.py` for reproducible single-image runs
-   `evaluate_dataset.py` for multiple-image evaluation
-   `app.py` for interactive visual inspection and quick experimentation

## Run locally on Windows

These commands are written for **PowerShell**, the usual terminal in VS
Code on Windows. Run them from the project root---the folder containing
`detect.py` and `requirements.txt`.

### 1. Open the project folder

In VS Code, select **Terminal → New Terminal** and change to the project
directory. For example:

``` powershell
cd "C:\Users\YOUR_USERNAME\Downloads\CMFD_results_and_report_outline"
```

Replace the example path with the actual folder on your computer.

### 2. Create and activate a virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can either use a terminal with
permission to activate scripts or invoke the virtual environment's
Python directly as shown below.

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you get `ModuleNotFoundError: No module named 'cv2'`, it usually
means OpenCV is not installed in the Python environment running the
script. Activate `.venv` and run the install command again. To install
the basic packages directly if necessary:

``` powershell
python -m pip install opencv-python numpy
```

If the requirements file lists additional packages, install those too.
Avoid installing conflicting OpenCV packages into the same environment.

### 4. Verify OpenCV and SIFT

``` powershell
python -c "import cv2; print('OpenCV:', cv2.__version__); print('SIFT available:', hasattr(cv2, 'SIFT_create'))"
```

The last value should be `True`. If it is `False`, check the installed
OpenCV version and the project's requirements.

### 5. Check the dataset paths

List matching forged-image and mask files recursively:

``` powershell
Get-ChildItem .\comofod_data -Recurse -File |
  Where-Object { $_.Name -match '001_[FB]\.' } |
  Select-Object FullName
```

This is only an example for ID `001`. Use the actual filenames and paths
shown by the command. If the files are nested, include the subfolder
names in the detector command.

### 6. Run detection on one forged image

If the images are directly inside `comofod_data`, an example command is:

``` powershell
python detect.py "comofod_data/001_F.png" --mask "comofod_data/001_B.png" --output "outputs/demo_001"
```

This example assumes the filenames and paths shown exist. Replace them
with your actual paths. The `--mask` argument is for evaluation/visual
comparison where supported; it does not provide the detector with the
ground-truth region for finding matches.

PowerShell supports multiline commands using a backtick (`` ` ``), not a
backslash:

``` powershell
python detect.py "comofod_data/001_F.png" `
  --mask "comofod_data/001_B.png" `
  --output "outputs/demo_001"
```

There must be no trailing spaces after a PowerShell continuation
backtick. The one-line command is simpler and less error-prone.

If you want to use the browser-based UI instead of the terminal, run:

``` powershell
streamlit run app.py
```

Then upload an image from the browser to test the same detection pipeline
interactively.

### 7. Inspect the output

Open the files generated under `outputs/demo_001`. Typical outputs
include:

-   `overlay.png`: input image with the predicted suspicious region
    highlighted.
-   `predicted_mask.png`: binary predicted region mask.
-   `inlier_matches.jpg`: geometrically consistent feature matches, if
    enough are available.
-   `result.json`: detector metadata and match/inlier information.

Output filenames depend on the version of the starter code. If a file is
not created, inspect the terminal output and the code's actual
output-writing logic.

## Run in Google Colab

A prepared notebook is available as `CMFD_Project_Google_Colab.ipynb` if
you are using the supplied Colab workflow.

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload the `.ipynb` notebook, then save a copy to Drive.
3.  Run the notebook cells in order.
4.  Upload `cmfd_course_project.zip` when prompted by the notebook.
5.  Install dependencies in the notebook.
6.  Download CoMoFoD from its official page if the notebook's direct
download fails; use the notebook's manual upload cell.
7.  Inspect the original/forged/mask examples, run single-image
detection, and then run the dataset evaluator.
8.  Run parameter experiments on development IDs and use a separate
    held-out subset for the final evaluation.
9.  Save generated outputs to Google Drive before ending the session.

Colab's `/content` runtime storage is temporary. Keep your notebook and
final generated results in Drive. If the runtime resets, the dataset may
need to be uploaded or extracted again. Do not add the dataset itself to
the results ZIP.

## Command-line usage

The basic command is:

``` powershell
python detect.py "PATH_TO_FORGED_IMAGE" --output "outputs/experiment_name"
```

If a matching ground-truth mask is available and the script supports the
`--mask` argument:

``` powershell
python detect.py "PATH_TO_FORGED_IMAGE" --mask "PATH_TO_GROUND_TRUTH_MASK" --output "outputs/experiment_name"
```

The project can be used in two modes:

-   Command-line mode: `python detect.py ...`
-   Interactive app mode: `streamlit run app.py`

`app.py` is the GUI wrapper around the detector and is intended for
manual testing and visual inspection. The actual algorithmic logic remains
in `detect.py`.

The starter implementation exposes parameter flags for experiments.
Confirm the supported options for your checked-out version with:

``` powershell
python detect.py --help
python evaluate_dataset.py --help
```

Parameters used by the supplied workflow include:

  -----------------------------------------------------------------------
  Parameter               What it controls        General trade-off
  ----------------------- ----------------------- -----------------------
  `--nfeatures`           Maximum number of SIFT  More features can help
                          features requested      on textured images but
                                                  increase runtime and
                                                  may add ambiguous
                                                  matches.

  `--min-distance`        Minimum spatial         A larger separation may
                          separation used to      reduce trivial local
                          reject unsuitable       matches but can exclude
                          matches, according to   genuine matches in
                          the implementation      nearby regions.

  `--ratio`               Lowe ratio-test         Lower is stricter and
                          threshold               often reduces false
                                                  matches, but may miss
                                                  genuine matches.

  `--ransac-threshold`    Geometric reprojection  A larger tolerance
                          tolerance for RANSAC    accepts more matches
                                                  but can admit incorrect
                                                  geometry; a smaller
                                                  tolerance can reject
                                                  valid matches.

  `--min-inliers`         Minimum consistent      Raising it can reduce
                          matches required by the false alarms while
                          detector                increasing missed
                                                  detections.

  `--dilation-size`       Size of the             Larger values can cover
                          morphological dilation  more of the copied area
                          used to expand the      but may overestimate
                          predicted mask          its boundaries.
  -----------------------------------------------------------------------

The exact parameter defaults should be read from your version of
`detect.py` or its `--help` output and reported accurately.

## Evaluation and experiments

Run the dataset evaluator to evaluate image-level detection and, where
matching ground-truth masks exist, pixel-level localisation. Example
usage for a dataset root is:

``` powershell
python evaluate_dataset.py "comofod_data" --output "outputs/evaluation"
```

The precise options supported by your version can be checked using
`python evaluate_dataset.py --help`. In the supplied workflow, options
may include limiting the number of image IDs or specifying a
comma-separated list of IDs.

### Recommended experiment protocol

1.  **Smoke test:** evaluate a small number of IDs first to check paths,
    file naming, and mask reading.
2.  **Development set:** choose a small, fixed set of image IDs for
    parameter tuning.
3.  **One-factor-at-a-time experiments:** change one parameter per
    experiment and keep the same development IDs.
4.  **Inspect images:** compare overlays, predicted masks, ground-truth
    masks, and false alarms on original images.
5.  **Held-out evaluation:** choose final parameters based on
    development results, then evaluate once on a disjoint set of IDs.
6.  **Save the configuration:** record the final parameter values, IDs,
    dataset category, and software versions.

Do not select parameters based on the held-out test scores and then
report those same scores as an unbiased final evaluation. If the dataset
is small, clearly state that results are preliminary and may not
generalise.

### Metrics

-   **Image-level accuracy:** proportion of images correctly classified
    as forged or original.
-   **Image-level precision:** proportion of images flagged as forged
    that are actually forged.
-   **Image-level recall:** proportion of forged images correctly
    detected.
-   **F1-score:** harmonic mean of precision and recall.
-   **Pixel precision and recall:** measure pixel-wise mask
    classification against the ground-truth mask.
-   **Intersection over Union (IoU):** overlap between predicted and
    ground-truth positive regions divided by their union.

For masks, the convention for which pixel value means "manipulated" must
be consistent between the ground truth and prediction. Check sample
masks visually. A convention mismatch can make pixel-level metrics
misleading.

## Output files

Depending on the command and version, the project may generate:

  ------------------------------------------------------------------------
  Output                               Purpose
  ------------------------------------ -----------------------------------
  `overlay.png`                        Visual explanation of the predicted
                                       suspicious area.

  `predicted_mask.png`                 Binary mask of the region estimated
                                       by the algorithm.

  `inlier_matches.jpg`                 Feature correspondences that
                                       survived geometric verification, if
                                       available.

  `result.json`                        Per-image detector metadata.

  `per_image_results.csv`              Per-image predictions and metric
                                       values from evaluation.

  `summary.json`                       Aggregate evaluation results.

  `development_experiment_table.csv`   Comparison of parameter settings on
                                       development IDs, if generated by
                                       the notebook.

  `heldout_final_results.csv`          Final held-out evaluation table, if
                                       generated by the notebook.

  `report_figure_*.png`                Figures generated for the report.
  ------------------------------------------------------------------------

Use actual outputs from your own run. Do not invent scores, imply that a
figure came from an experiment if it did not, or describe a predicted
mask as ground truth.

## How to improve and compare results

Use a controlled experiment table in the report. For each run, record
the IDs, parameter values, runtime if measured, image-level
precision/recall/F1, and pixel-level F1/IoU when available.

A suggested table:

  -------------------------------------------------------------------------------------------------------------------
  Experiment   Changed              Image   Image recall       Image F1       Pixel F1            IoU Observations
               parameter        precision                                                             
  ------------ ----------- -------------- -------------- -------------- -------------- -------------- ---------------
  Baseline     None          \[measured\]   \[measured\]   \[measured\]   \[measured\]   \[measured\] \[actual
                                                                                                      observation\]

  Experiment 1 Ratio         \[measured\]   \[measured\]   \[measured\]   \[measured\]   \[measured\] \[actual
               threshold                                                                              observation\]

  Experiment 2 Minimum       \[measured\]   \[measured\]   \[measured\]   \[measured\]   \[measured\] \[actual
               inliers                                                                                observation\]

  Experiment 3 Dilation      \[measured\]   \[measured\]   \[measured\]   \[measured\]   \[measured\] \[actual
               size                                                                                   observation\]
  -------------------------------------------------------------------------------------------------------------------

Recommended questions to investigate:

-   Does a stricter ratio test improve precision at the cost of recall?
-   Does increasing the minimum inlier count reduce false positives on
    original images?
-   Does increasing dilation improve mask coverage while making
    boundaries less accurate?
-   Which image types fail most often: low-texture images, repetitive
    textures, small copied regions, or strongly transformed regions?

A parameter change is not automatically an improvement. Report the
observed trade-offs, including negative or inconclusive results.

## Results

**Complete this section after running the code. Do not leave example
values as if they were measured results.**

-   Dataset category and source: \[TODO\]
-   Number of image IDs evaluated: \[TODO\]
-   Number of individual images evaluated: \[TODO\]
-   Development IDs: \[TODO\]
-   Held-out test IDs: \[TODO\]
-   Final parameter settings: \[TODO\]
-   Image-level precision / recall / F1: \[TODO\]
-   Pixel-level precision / recall / F1 / IoU: \[TODO, if available\]
-   Best qualitative example: \[TODO: image ID and reason\]
-   Failure case: \[TODO: image ID and reason\]

Include at least one real forged-image example with its ground-truth
mask, predicted mask, and overlay. Include a failure case or false
positive if available. Explain what the results do and do not establish.

## Limitations

-   Repeated natural patterns (e.g., windows, tiles, foliage, bricks)
    can create false matches.
-   Low-texture or blurred regions may not contain enough reliable SIFT
    keypoints.
-   Strong scaling, rotation, compression, noise, or other
    postprocessing may reduce match quality.
-   The geometric model may not fit every real-world transformation.
-   A convex hull and dilation provide approximate localisation and can
    include unmanipulated pixels.
-   Thresholds that work on one dataset subset may not generalise to
    other images or manipulation methods.
-   Copy-move detection alone does not detect all kinds of image
    forgery.
-   Automated results should be treated as cues for further inspection,
    not as proof of manipulation.


## References and dataset acknowledgement

1.  Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013). "CoMoFoD
    --- New Database for Copy-Move Forgery Detection." *Proceedings of
    the 55th International Symposium ELMAR*, pp. 49--54. See the
    [official CoMoFoD website](https://www.vcl.fer.hr/comofod/).
2.  Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant
    Keypoints." *International Journal of Computer Vision*, 60, 91--110.
3.  Fischler, M. A., & Bolles, R. C. (1981). "Random Sample Consensus: A
    Paradigm for Model Fitting with Applications to Image Analysis and
    Automated Cartography." *Communications of the ACM*, 24(6),
    381--395.
4.  OpenCV documentation: [Feature matching](https://docs.opencv.org/).
5.  CoMoFoD [release
    agreement](https://www.vcl.fer.hr/comofod/release_agreemen.html).
    Review and follow the conditions of use.
