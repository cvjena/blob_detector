# C++ Version of the Blob Detector

## Installation

### Using Anaconda / Miniconda
```bash
conda create -n blob_detector -y python~=3.9.0 opencv~=4.5
conda activate blob_detector
```

### Compilation and installation
```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
make -j && make install
```
