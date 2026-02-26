# Hierarchical Feedback Model for Occluded Face Processing

A computational model for occluded face recognition based on VTC-vlPFC hierarchical feedback mechanism

---

## Overview

This model implements hierarchical feedback interactions between ventral temporal cortex (VTC) and ventrolateral prefrontal cortex (vlPFC) for processing occluded faces. The model combines:
- **Self-Organizing Maps (SOM)** - Implementing topographic organization
- **Stochastic Hopfield Networks** - Implementing attractor dynamics
- **Top-down Feedback** - vlPFC modulation of VTC

The model is validated under 5 occlusion conditions: intact faces, eyes-occluded, upper-half, lower-half, and eyes-only.

---

## Core Algorithm Pipeline

### 1. Input Processing
```
Raw Image (224×224)
  → AlexNet
  → PCA Dimensionality Reduction (4D)
  → Normalization
```

### 2. VTC Self-Organizing Map
```python
# VTC SOM (200×200 neurons)
Input Vector (4D) → SOM Activation → Topographic Response Map (200×200)
```
- Using Gaussian neighborhood function
- Weight normalization
- Forward activation: activation = dot(weights, input)

### 3. Hopfield Network Dynamics

#### Mode A: Recurrent-only Mode
```
VTC Initial State → VTC Recurrent Dynamics → VTC Stable State
```

#### Mode B: Feedback Mode
```
VTC Initial State
  ↓
VTC Recurrent Dynamics
  ↓
vlPFC Activation (from VTC compressed representation)
  ↓
vlPFC Recurrent Dynamics
  ↓
vlPFC→VTC Feedback Projection
  ↓
VTC Continues Evolution under Feedback Modulation
  ↓
VTC Stable State
```

### 4. Stochastic Update Rule
```python
# For each neuron i
local_field = Σ(w_ij * s_j) + H_external
```
- `H_external`: External field (bottom-up input + top-down feedback)
- Asynchronous update: randomly select one neuron to update at each step

### 5. Energy Function
```
E = -0.5 * Σ(w_ij * s_i * s_j)
```
- Network tends to minimize energy
- Stable states correspond to energy minima (attractors)

---

## System Requirements

- **Python**: 3.8+
- **Memory**: 16 GB minimum, 32 GB recommended
- **Storage**: 20 GB
- **GPU**: Optional (only for AlexNet feature extraction)

---

## Installation

### 1. Create Environment
```bash
conda create -n occluded_face python=3.8
conda activate occluded_face
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Weights
Download the following files and place them in the `Formal/` directory:
- `som_sigma_6.2.npy` - VTC SOM weights (200×200×4)
- `model_VTC_weights.npy` - VTC Hopfield weights (40000×40000, ~12GB)
- `som_vlPFC_weights.npy` - vlPFC SOM weights (20×20×40000)
- `model_vlPFC_weights.npy` - vlPFC Hopfield weights (400×400)
- `face_mask.npy` - Face-selective region mask
- `object_mask.npy` - Object-selective region mask
- `Data.npy` - AlexNet feature data
- `mean.npy`, `std.npy` - Normalization parameters

**⚠️ Important:** Extract the contents from the Release package and place them in the directory according to the File Structure format (see below) before running the code.

---

## Usage

### Quick Start

#### 1. Load Models
```python
import numpy as np
import BrainSOM
import Hopfield_VTCSOM

# Load VTC SOM
som_VTC = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1)
som_VTC._weights = np.load('som_sigma_6.2.npy')

# Load VTC Hopfield Network
model_VTC = Hopfield_VTCSOM.Stochastic_Hopfield_nn(
    x=200, y=200, pflag=1, nflag=-1,
    patterns=[None, None, None, None]
)
model_VTC._w = np.load('model_VTC_weights.npy')

# Load vlPFC Models (required for feedback mode)
som_vlPFC = BrainSOM.VTCSOM(20, 20, 40000, sigma=6, learning_rate=1)
som_vlPFC._weights = np.load('som_vlPFC_weights.npy')

model_vlPFC = Hopfield_VTCSOM.Stochastic_Hopfield_nn(
    x=20, y=20, pflag=1, nflag=-1,
    patterns=[None, None, None, None]
)
model_vlPFC._w = np.load('model_vlPFC_weights.npy')
```

#### 2. Run Feedback Mode
See complete implementation in `Occluded_face_formal.ipynb`.

### Main Notebook Files

1. **Stimuli_formal.ipynb** - Stimulus Preparation
   - AlexNet feature extraction
   - GradCAM attention analysis
   - Feature normalization

2. **Occluded_face_formal.ipynb** - Model Simulation
   - VTC-vlPFC hierarchical dynamics
   - Simulation of 5 occlusion conditions
   - Results saved to `model_results/`

3. **Model_results_formal.ipynb** - Results Analysis
   - Temporal dynamics visualization
   - Energy landscape analysis (UMAP)
   - Decoding analysis
   - Manifold geometry metrics

---

## Key Parameters

### SOM Parameters
- `sigma`: Neighborhood width
- `learning_rate`: Learning rate (typically 1.0)
- `neighborhood_function`: Neighborhood function type ('gaussian')

### Hopfield Parameters
- `beta`: Inverse temperature, controls stochasticity
  - High values (100-200): More deterministic, fast convergence
  - Low values (10-50): More stochastic, explores more states
- `epochs`: Number of update steps
  - Recurrent-only: 80,000 steps
  - Feedback: 80,000 steps (vlPFC starts at 50,000)
- `save_inter_step`: Save interval (1000 steps)

### Feedback Parameters
- `top_down_strength`: Feedback strength (default: 4)
- `vlPFC_start_time`: vlPFC start time (default: 50,000)

---

## Output Results

### 1. Dynamic State Trajectories
- `Dynamic_states_VTC`: VTC activity over time (20 images × 81 timepoints × 200 × 200)
- `Dynamic_states_vlPFC`: vlPFC activity over time (20 images × 31 timepoints × 20 × 20)

### 2. Energy Landscape
- UMAP-reduced network states
- Energy surface visualization
- Attractor basin analysis

### 3. Performance Metrics
- **Decoding Accuracy**: Face vs. tool classification (5 conditions × 20 samples)
- **Manifold Dimensionality**: Intrinsic dimensionality estimation
- **Manifold Radius**: Geometric spread of representations

---

## File Structure

```
Formal/
├── BrainSOM.py                    # SOM implementation
├── Hopfield_VTCSOM.py             # Stochastic Hopfield network
├── Stimuli_formal.ipynb           # Stimulus preparation
├── Occluded_face_formal.ipynb     # Main simulation
├── Model_results_formal.ipynb     # Results visualization
├── Stim_for_model/                # Input stimuli (20 images per condition)
│   ├── face/                      # Intact faces
│   ├── noeye/                     # Eyes-occluded
│   ├── top_face/                  # Upper-half
│   ├── down_face/                 # Lower-half
│   └── eyes/                      # Eyes-only
├── model_results/                 # Output directory
├── *.npy                          # Pre-trained weights
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
