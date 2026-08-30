---
language: en
tags:
- PyTorch
- computer-vision
- remote-sensing
- landslide-detection
license: mit
---

# Multimodal Landslide Forecasting Model (Sri Lanka)

This repository contains the trained weights and architecture for a Multimodal Late-Fusion model used to predict landslides from Sentinel-2 and SRTM satellite imagery.

## Model Details
*   **Architecture:** ResNet50 (RGB) + Custom CNN (Terrain - NIR, DEM, Slope)
*   **Performance:** 
    * Recall: 91%
    * ROC-AUC: 0.7371
    * Optimal Threshold: 0.291
*   **Dataset:** Ditwah Cyclonic Storm (2025) Landslides provided by Arthur C. Clarke Institute for Modern Technologies.

## Usage
The model accepts a 6-channel tensor of shape `(6, 224, 224)`.
Channels: `[B4, B3, B2, B8, DEM, Slope]`.
See `10_train_multimodal_pytorch.py` for the PyTorch implementation.
