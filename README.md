# Blob Detector

Clone me with `git clone git@github.com:cvjena/blob_detector.git`

## Description
Blob detection algorithms for insects on a single-color (white) screen.

## Installation

### Via `pip`
```bash
pip install ammod-blob-detector
```

### From Source
```bash
conda create -n detector python~=3.9.0
conda activate detector
pip install --upgrade pip
pip install -r requirements.txt
make
```

## Example script
```bash
# will require to install additional packages: pip install cvargparse~=0.5 pyqt5 scikit-image
python blob_detector/detect.py blob_detector_cpp/examples/images/2021-08-02_Weinschale_4846.JPG
```

## C++ Implementation
*See [blob_detector_cpp](blob_detector_cpp) for more details.*


## Licence
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
