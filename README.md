# Low-Dimensional Frontal Feedback Resolves High-Dimensional Visual Ambiguity in Human Visual Cortex

This repository contains the code and bundled data used for the paper *Low-Dimensional Frontal Feedback Resolves High-Dimensional Visual Ambiguity in Human Visual Cortex*. The computational model implements hierarchical interactions between a ventral temporal cortex (VTC) module and a ventrolateral prefrontal cortex (vlPFC) module to study occluded face processing.

The release includes:

- model code: `BrainSOM.py`, `Hopfield_VTCSOM.py`
- stimulus folders for six conditions: intact face, no-eyes, upper-half, lower-half, eyes-only, and tools
- pretrained model weights and normalization statistics
- precomputed model outputs in `model_results/`
- analysis notebooks used to reproduce the main computational figures

The repository is notebook-centered. The main entry points are:

- `Stimuli_formal.ipynb`: stimulus preparation and Grad-CAM based face-information estimation
- `Occluded_face_formal.ipynb`: hierarchical VTC-vlPFC simulation
- `Model_results_formal.ipynb`: manifold, decoding, and time-series analysis

## Overview

The model follows the paper's computational pipeline:

1. Each image is resized/cropped to `224 x 224`, processed by AlexNet, and represented in a `1000`-dimensional feature space.
2. The feature vector is normalized and projected to the first `4` PCA components.
3. The `4`-dimensional feature drives a `200 x 200` VTC self-organizing map.
4. VTC activity evolves under stochastic Hopfield dynamics.
5. In the feedback model, the current VTC state is projected to a `20 x 20` vlPFC SOM, updated by vlPFC recurrent dynamics, and projected back to VTC as top-down feedback.
6. The code stores VTC and vlPFC state trajectories, which are then analyzed in the paper through decoding, manifold geometry, and energy-landscape visualization.

The provided stimuli implement the paper's Information-Graded Occluded Faces (IGOF) design:

- `face`: intact faces
- `noeye`: eyes occluded
- `top_face`: upper-half face
- `down_face`: lower-half face
- `eyes`: eyes-only
- `tools`: non-face comparison category

Each folder contains `20` images.

## System Requirements

### Hardware

The repository is large and memory-intensive.

- CPU: standard 64-bit desktop CPU
- RAM:
  - `32 GB` recommended for the precomputed-results demo in `Model_results_formal.ipynb`
  - `64 GB` recommended for rerunning the full VTC-vlPFC simulation in `Occluded_face_formal.ipynb`
- Storage:
  - the current `Formal/` folder occupies about `26.4 GB`
  - reserve at least `40 GB` free disk space to run the notebooks comfortably
- GPU: optional; helpful for AlexNet-based feature extraction and Grad-CAM, but not required for the bundled-results demo

### Software

The code was checked in the author's conda environment `occluded_face` with:

- OS: Windows 11 24H2, 64-bit (`build 26100`; reported by Python as `Windows-10-10.0.26100-SP0`)
- Python: `3.9.21`

Primary Python packages used by the notebooks and source code:

- `numpy==1.23.5`
- `scipy==1.9.3`
- `pandas==2.2.3`
- `torch==2.7.1`
- `torchvision==0.22.1`
- `scikit-learn==1.6.1`
- `matplotlib==3.9.4`
- `seaborn==0.13.2`
- `Pillow==11.2.1`
- `opencv-python==4.11.0.86`
- `imageio==2.37.0`
- `umap-learn==0.5.7`
- `statsmodels==0.14.4`
- `patsy==1.0.1`
- `tqdm==4.67.1`
- `h5py==3.13.0`
- `joblib==1.5.1`
- `minisom==2.3.5`
- `dhnn==0.1.12`
- `ipykernel==6.29.5`
- `ipywidgets==8.1.7`

Notes:

- `requirements.txt` is included, but the code also requires `minisom` and `dhnn`, which are imported by `BrainSOM.py` and `Hopfield_VTCSOM.py`.
- If you want to execute the notebooks from the command line, install `notebook` or `jupyter` as well.

## Installation Guide

### Recommended installation

```bash
conda create -n occluded_face python=3.9.21 -y
conda activate occluded_face
python -m pip install -r requirements.txt
python -m pip install minisom==2.3.5 dhnn==0.1.12 notebook ipywidgets
```

If you need a CUDA-enabled PyTorch build, install PyTorch and torchvision from the official PyTorch channel first, then install the remaining packages.

### Typical install time

Typical install time on an ordinary desktop computer with a stable internet connection: `~20-30 minutes`.

### Files already included in this release

No additional download is required for the bundled demo below. This release already contains:

- pretrained weights such as `model_VTC_weights.npy`, `model_vlPFC_weights.npy`, `som_sigma_6.2.npy`, `som_vlPFC_weights.npy`
- normalization statistics: `mean.npy`, `std.npy`
- PCA fitting data: `Data.npy`
- precomputed demo outputs in `model_results/`

### Additional download for large files

Due to GitHub file size limits, some large files cannot be uploaded directly to the repository. They are provided here:

`https://cloud.tsinghua.edu.cn/library/c46fb051-ac22-4cbf-a484-de0c3b08c151/Occluded_face/`

Please download the required compressed files from that link, extract them, and place the extracted contents into the repository according to the directory structure shown at the end of this `README.md`.

## Demo

The demo and analysis code are contained in the three notebooks below:

- `Stimuli_formal.ipynb`
- `Occluded_face_formal.ipynb`
- `Model_results_formal.ipynb`

The expected outputs are shown directly in the corresponding notebook cells.

### Expected demo run time

Expected run time on an ordinary desktop computer with `32 GB` RAM and an SSD: `~10-20 minutes`.

Most of this time is spent loading the bundled `model_results/` files, which occupy about `13 GB` in total.

## Instructions for Use

### Using the provided stimuli

`Occluded_face_formal.ipynb` is the main notebook for running the model on image folders. The notebook:

1. loads AlexNet
2. fits PCA on `Data.npy`
3. loads the pretrained SOM and Hopfield weights
4. converts each image into a `4`-dimensional PCA feature
5. runs the VTC-only or VTC-vlPFC stochastic dynamics
6. saves the outputs as Python dictionaries containing dynamic states and feedback terms

The notebook currently points to:

- input root: `Stim_for_model/`
- default output root: `VTC_vlPFC_model/`

If you want the saved files to be read directly by `Model_results_formal.ipynb`, save them with the same naming convention used in `model_results/`, for example:

- `Face_feedback_results.npy`
- `Top_face_feedback_results.npy`
- `Noeye_feedback_results.npy`
- `Down_face_feedback_results.npy`
- `Eyes_feedback_results.npy`
- `Tool_feedback_results.npy`

### Using your own data

To run the model on your own images:

1. Create a new folder under `Stim_for_model/`, for example `Stim_for_model/my_condition/`.
2. Put your `.png`, `.jpg`, or `.bmp` images into that folder.
3. Open `Occluded_face_formal.ipynb`.
4. Keep the same preprocessing used by the paper:
   - resize to `256`
   - center crop to `224 x 224`
   - normalize with ImageNet mean/std inside the notebook
   - extract AlexNet features
   - z-score/normalize with the bundled `mean.npy` and `std.npy`
   - project to the first four PCA components fitted from `Data.npy`
5. Replace the input path in the notebook, or call the helper already used there:

```python
images_response, Dynamic_states_VTC, Dynamic_states_vlPFC, F_all, H_top_down_all = Feedback_results(
    'Stim_for_model/my_condition/', mean, std
)
```

6. Save the output dictionary with `pickle.dump(...)` using the same keys as the bundled files:
   - `Dynamic_states_VTC`
   - `Dynamic_states_vlPFC`
   - `F`
   - `H_top_down`

## Repository Contents

```text
Formal/
|-- BrainSOM.py
|-- Hopfield_VTCSOM.py
|-- Stimuli_formal.ipynb
|-- Occluded_face_formal.ipynb
|-- Model_results_formal.ipynb
|-- Stim_for_model/
|   |-- face/
|   |-- noeye/
|   |-- top_face/
|   |-- down_face/
|   |-- eyes/
|   `-- tools/
|-- model_results/
|-- Transfer_AexNet
|-- Data.npy
|-- mean.npy
|-- std.npy
|-- face_mask.npy
|-- object_mask.npy
|-- som_sigma_6.2.npy
|-- som_vlPFC_weights.npy
|-- model_VTC_weights.npy
|-- model_vlPFC_weights.npy
|-- X_VTC.npy
|-- X_vlPFC.npy
|-- H_VTC_recurrent.npy
|-- H_VTC_feedback.npy
|-- H_vlPFC_feedback.npy
|-- model_dimensions_VTC.npy
|-- model_dimensions_vlPFC.npy
|-- model_radii_VTC.npy
|-- model_radii_vlPFC.npy
|-- LICENSE
|-- requirements.txt
`-- README.md
```

## License

This project is released under the MIT License. See `LICENSE`.
