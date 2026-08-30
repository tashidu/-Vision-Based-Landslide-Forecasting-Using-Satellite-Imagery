# Vision-Based Landslide Forecasting Using Satellite Imagery

This repository contains the source code, experimental notebooks, and documentation for the Vision-Based Systems Group Assessment on Landslide Forecasting. 

The project applies modern Deep Learning and Computer Vision techniques to multi-channel satellite imagery to predict and forecast landslide-prone areas.

## 🏔️ Project Overview

During the Ditwah Cyclonic Storm in 2025, Sri Lanka experienced widespread slope failures. This experimental academic system attempts to classify geographical regions as vulnerable to landslides using satellite imagery, serving as an early-warning prototype.

We utilize a **Multimodal Late-Fusion Architecture**, combining:
1. **Visual Features (RGB):** Processed via a pre-trained **ResNet50** CNN.
2. **Physical Terrain Features (NIR, Elevation, Slope):** Processed via a custom **Terrain CNN**.

## 📊 Dataset & Acknowledgement

The landslide boundary demarcation dataset was provided by the **Arthur C. Clarke Institute for Modern Technologies**, the nationally mandated institution for space science activities in Sri Lanka. 
*Prepared by: Mahesh Chathurange & W.G.N.N. Jayawardhana (Space Applications Division).*

### Satellite Data Sourcing
We dynamically obtained corresponding satellite imagery using **Google Earth Engine (GEE)**:
*   **Sentinel-2 (Level-2A):** B4 (Red), B3 (Green), B2 (Blue), B8 (Near-Infrared).
*   **SRTM:** Digital Elevation Model (DEM) and derived Slope Gradient.

*(Note: Actual datasets and raw numerical `.npy` files are excluded from this repository due to size constraints. See `.gitignore`)*

## 🚀 Repository Structure

### Core Scripts & Notebooks
*   `06_bulk_download_v2.py`: Initial script for downloading visual Sentinel-2 patches.
*   `09_download_proposed_6ch.py`: Upgraded script for downloading the 6-channel Multimodal patches (RGB + NIR + DEM + Slope).
*   `08_train_resnet50_pytorch_new.ipynb`: Experiment 2 - Transfer Learning with ResNet50 on visual data.
*   `10_train_multimodal_pytorch.py`: Experiment 3 - Final Multimodal Late-Fusion architecture and training logic.

### Documentation & Reports
*   `baseline_documentation.md`: Strategy and results for the baseline CNN.
*   `resnet50_methodology_and_results.md`: Detailed report on the ResNet50 implementation and optimal thresholding.
*   `final_experiments_conclusion.md`: Comparative analysis of all models.

### Continuous-Use System (UI)
*   `app.py`: A fully interactive **Streamlit dashboard** that allows users to select patches, view physical/visual colormaps, and run live model inference.

## 🛠️ Installation and Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tashidu/-Vision-Based-Landslide-Forecasting-Using-Satellite-Imagery.git
   ```
2. **Set up Virtual Environment & Install Dependencies:**
   Ensure you have Python 3.12 installed.
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   pip install torch torchvision pandas numpy matplotlib earthengine-api streamlit plotly
   ```
3. **Run the Streamlit Dashboard (UI):**
   ```bash
   streamlit run app.py
   ```

## 📈 Performance Results

Our final **Multimodal Model** achieved the following on a geographically-isolated test set:
*   **Recall (Landslides Found):** 91%
*   **ROC-AUC:** 0.7371
*   **Optimal Threshold:** 0.291

By prioritizing a lower threshold, the system acts as a highly sensitive early-warning radar, catching almost all actual landslides at the acceptable cost of a higher false-positive rate.

## ⚠️ Important Scientific and Ethical Considerations

This is strictly an **experimental academic system**. Satellite imagery alone is insufficient for operational disaster warnings. A real-world deployment requires integration with real-time rainfall sensors, soil moisture data, geological mapping, and validation by certified geologists. **Do not use this system to create public alarm or make unsupported safety claims.**
