# Dataset

## Dataset used in this project

**Name:** CoMoFoD --- Copy-Move Forgery Detection Dataset

**Official dataset download page:**\
https://www.vcl.fer.hr/comofod/download.html

**Official project website:**\
https://www.vcl.fer.hr/comofod/

**Dataset release agreement:**\
https://www.vcl.fer.hr/comofod/release_agreemen.html

## Which part of the dataset to use

For the starter workflow, download the **small-image** dataset category
from the official download page, if available. The project uses matching
original, forged, and binary-mask images where provided:

-   `*_O.*` --- original image
-   `*_F.*` --- forged image
-   `*_B.*` --- binary ground-truth mask

The exact filenames, extensions, and folder structure can vary by
download package, so check the extracted files before running the code.

## Important usage note

Read and comply with the dataset's official release agreement before
downloading or using the files. Do not upload the dataset archive or
image files to this repository unless the official terms explicitly
allow redistribution. Cite the CoMoFoD dataset paper in your project
report.

**Dataset paper:**\
Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013). *CoMoFoD ---
New Database for Copy-Move Forgery Detection*. Proceedings of the 55th
International Symposium ELMAR, pp. 49--54.
