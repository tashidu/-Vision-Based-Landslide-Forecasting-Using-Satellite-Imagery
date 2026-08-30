# Vision-Based Landslide Forecasting Using Satellite Imagery
**Module:** Vision-Based Systems  
**Project:** Group Assessment  

---

## 1. Background and Motivation
Landslides are among the most destructive natural hazards in Sri Lanka. During the Ditwah Cyclonic Storm in 2025, prolonged rainfall resulted in widespread landslides across the country. A reliable landslide forecasting system can support early identification of vulnerable locations, aiding disaster preparedness and resource allocation. 

This project explores the application of deep learning and computer vision to satellite imagery to distinguish visual patterns associated with landslide occurrences.

---

## 2. Dataset Strategy & Acquisition
### 2.1 Dataset Definition
We utilized a dataset containing geographical polygons of over 4,000 mass movements associated with the Ditwah Cyclonic Storm.

### 2.2 Satellite-Image Services Used
We utilized **Google Earth Engine (GEE)** to programmatically acquire multi-channel satellite data corresponding to the landslide coordinates.

*   **Sentinel-2 (Level-2A Surface Reflectance):** Used to acquire pre-event optical and near-infrared bands. We filtered for images between Jan-Oct 2025 with <20% cloud cover.
*   **SRTM (Shuttle Radar Topography Mission):** Used to acquire Digital Elevation Model (DEM) data, from which we also derived Slope gradient.

### 2.3 Image Acquisition & Preprocessing
For each coordinate, we exported a `224x224` pixel bounding box.
We extracted 6 distinct channels of physical data, saved as `.npy` arrays for precision:
1.  **B4 (Red)** - Visible spectrum
2.  **B3 (Green)** - Visible spectrum
3.  **B2 (Blue)** - Visible spectrum
4.  **B8 (Near-Infrared / NIR)** - Critical for detecting vegetation stress and landslide scars.
5.  **DEM (Elevation)** - Absolute altitude in meters.
6.  **Slope** - Steepness of the terrain in degrees.

**Data Split (Avoiding Leakage):**
To prevent data leakage, we grouped the dataset by distinct geographical landslide events (using `GroupShuffleSplit`), ensuring patches from the same landslide cluster were not mixed between the Train (70%), Validation (15%), and Test (15%) sets.

### 2.4 Dataset Acknowledgement
The landslide boundary demarcation dataset was prepared by the **Arthur C. Clarke Institute for Modern Technologies**, the nationally mandated institution for space science activities in Sri Lanka. 
*Prepared by: Mahesh Chathurange & W.G.N.N. Jayawardhana (Space Applications Division).*

---

## 3. Model Development
We conducted three progressive experiments to build the forecasting model:

### 3.1 Stage 1: Baseline CNN
We first developed a simple 4-layer Convolutional Neural Network (CNN) trained solely on RGB images (B4, B3, B2). 
*   **Purpose:** To establish a performance floor.
*   **Performance:** Achieved an ROC-AUC of **0.5982**.

### 3.2 Stage 2: Transfer Learning (ResNet50)
We upgraded the architecture to use a pre-trained **ResNet50** model (using ImageNet weights). We froze the early layers and fine-tuned the final convolutional block on our RGB data.
*   **Improvements:** Applied ImageNet standardization and RandomResizedCrop augmentation.
*   **Performance:** Achieved an ROC-AUC of **0.7135**. 
*   **Optimal Thresholding:** We shifted from a static 0.5 threshold to an optimal threshold (0.239) calculated via the validation ROC curve, drastically improving Recall to 97%.

### 3.3 Stage 3: Proposed Forecasting Model (Multimodal Late-Fusion)
Our final proposed system uses a **Multimodal Late-Fusion Architecture** that processes both visual (RGB) and physical (Terrain) data simultaneously.
*   **RGB Branch:** A pre-trained ResNet50 processes the B4, B3, B2 bands, extracting complex visual features (flattened to 2048 dimensions).
*   **Terrain Branch:** A custom 4-layer CNN processes the B8 (NIR), DEM, and Slope bands (flattened to 256 dimensions).
*   **Fusion Head:** The features (2304 dimensions) are concatenated and passed through a Multi-Layer Perceptron (MLP) with Dropout (0.5) and Batch Normalization to output a final Sigmoid probability.

---

## 4. Evaluation and Results

Our Multimodal model was evaluated on the strict geographically-separated test set.

**Final Test Metrics (Threshold = 0.291):**
*   **Accuracy:** 63%
*   **Precision (Landslide):** 0.59
*   **Recall (Landslide):** 0.91 (91%)
*   **Specificity (Safe):** 0.35 (35%)
*   **ROC-AUC:** **0.7371**

**Confusion Matrix:**
*   True Positives (Landslides correctly found): **570**
*   False Negatives (Landslides missed): **57**
*   True Negatives (Safe areas confirmed safe): **217**
*   False Positives (Safe areas flagged as danger): **403**

### 4.1 Practical Consequences & Error Analysis
*   **High Recall (91%) / False Positives:** We intentionally tuned the threshold to favor Recall. In disaster forecasting, a False Negative (missing a landslide) costs lives. A False Positive (unnecessary alert) is an acceptable trade-off, acting as a cautious early-warning radar.
*   **Cloud Contamination:** Despite filtering for <20% clouds, localized cloud cover in Sentinel-2 imagery remains a limitation that can mask ground features.
*   **Image Resolution:** Sentinel-2 provides 10m/pixel resolution. Small, micro-scale slope failures might not be fully captured by this spatial resolution compared to commercial drone imagery.

---

## 5. Continuous-Use System (Streamlit Prototype)
We developed a fully functional, interactive software system (`app.py`) for continuous model inference.
*   **Features:** Users can select testing patches from the dataset via a dropdown menu.
*   **Visual Explanations:** The UI dynamically renders the RGB True Color image alongside colormaps for NIR, Elevation, and Slope, allowing the user to visually inspect the physical parameters.
*   **Inference:** The dashboard runs the Multimodal PyTorch model in real-time, outputting the exact percentage probability and a clear safety/danger badge based on the user-adjustable threshold.

---

## 6. Scientific and Ethical Considerations
### Limitations & Operational Use
This model is presented strictly as an **experimental academic system**. It must not be used to create public alarm or make unsupported claims about the safety of locations.

Satellite imagery alone is insufficient for operational disaster warnings. A real-world deployment would require integration with:
*   Real-time rainfall measurements and live soil-moisture sensors.
*   Geological rock-type data and soil drainage maps.
*   Validation by certified geologists and disaster-management professionals.
