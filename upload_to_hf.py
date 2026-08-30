import os
from huggingface_hub import HfApi

token = "YOUR_HF_TOKEN" # Replace with your token if you run this manually
api = HfApi(token=token)

repo_id = "vinukatashidu/landslide-detection-multimodal"

print(f"Creating Hugging Face Model Repository: {repo_id}...")
# Create repo (Model type)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Generate README.md with Model Card frontmatter
readme_content = """---
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
"""
with open("hf_model_readme.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Uploading README.md (Model Card)...")
api.upload_file(path_or_fileobj="hf_model_readme.md", path_in_repo="README.md", repo_id=repo_id, repo_type="model")

print("Uploading 10_train_multimodal_pytorch.py...")
api.upload_file(path_or_fileobj="10_train_multimodal_pytorch.py", path_in_repo="10_train_multimodal_pytorch.py", repo_id=repo_id, repo_type="model")

print("Uploading model weights (multimodal_stage2.pth)...")
api.upload_file(path_or_fileobj="saved_models/multimodal_stage2.pth", path_in_repo="multimodal_stage2.pth", repo_id=repo_id, repo_type="model")

print("Upload complete! Your Model is now live.")
print(f"URL: https://huggingface.co/{repo_id}")
