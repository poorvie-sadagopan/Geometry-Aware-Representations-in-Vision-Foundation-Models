# Setup and Execution Guide

## 1. System Requirements & Dependencies
* **Framework:** PyTorch.
* **Hardware:** NVIDIA Tesla T4 GPUs (or equivalent) for hardware acceleration.
* **Libraries:** Standard scientific computing libraries for processing high-dimensional hidden states.
* **Pre-trained Backbones Required:** DINOv2, CLIP, DepthAnything V2, and ViT-Small.

## 2. Dataset Preparation
Download and configure the following datasets before running the experiments:
* **Hypersim:** Access via the partitioned repositories (Parts 1, 2, 3, and 4). Used for Phase 1 diagnostics and Phase 2 surface normal evaluation. Note: Due to large dataset size, chunk the training data into groups of 8 scenes per chunk to manage GPU memory.
* **NYU-Depth-v2:** Used for Phase 2 depth prediction evaluations.

## 3. Running Phase 1: Diagnostics
1. **Extract Hidden States:** Run the pre-trained backbones over the Hypersim dataset and extract hidden states from the individual layers.
2. **Linear Probing:** Train linear models on these hidden states to predict depth, camera space surface normals, and world space surface normals.
3. **CKA Computations:** Compute Centered Kernel Alignment (CKA) similarities between the layer predictions and ground truth, as well as pairwise CKA between internal layers.
4. **Layer Selection:** Select layers using the Gapped MMR combined score calculation. Ensure no two consecutive layers are selected.

## 4. Running Phase 2: Decoder Training

### A. Initial Depth Experiment
* **Dataset:** NYUv2 only.
* **Epochs:** 50.
* **Batch Size:** 8.
* **Optimizer:** AdamW.
* **Learning Rate:** 1e-4.
* **Weight Decay:** 1e-4.
* **Scheduler:** OneCycleLR (stepped per batch).
* **Loss Function:** Scale-invariant logarithmic loss (variance focus 0.85).

### B. Scaled Depth Experiment
* **Dataset:** NYUv2.
* **Epochs:** 50.
* **Optimizer:** AdamW.
* **Learning Rate & Weight Decay:** 1e-4 for both.
* **Scheduler:** OneCycleLR (stepped per batch).
* **Gradient Clipping:** Maximum norm 1.0.
* **Loss Function:** Scale-invariant logarithmic loss (variance focus 0.85).

### C. Surface Normal Experiment
* **Dataset:** Hypersim only (chunked into 8 scenes per sequential iteration).
* **Epochs:** 30.
* **Optimizer:** AdamW.
* **Learning Rate & Weight Decay:** 1e-4 for both.
* **Scheduler:** OneCycleLR (stepped per batch).
* **Gradient Clipping:** Maximum norm 1.0.
* **Loss Function:** Cosine distance loss.

### D. Data Augmentation (Applied to All Phase 2 Runs)
* **Spatial Flips:** Apply horizontal flip with a 0.5 probability. *Crucial:* For normal estimation, negate the x-component of the normal vector before flipping.
* **Color Jitter:** Apply only to input images, never to labels. 
    * Brightness, contrast, and saturation perturbations of +/- 20% (0.8 probability).
    * Hue perturbation of +/- 0.05 (0.2 probability).
