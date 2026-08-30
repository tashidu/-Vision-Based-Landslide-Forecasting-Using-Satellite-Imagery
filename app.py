import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
import plotly.graph_objects as go

st.set_page_config(page_title="Landslide Detection Model", layout="wide", page_icon="🏔️")

# --- Model Architecture ---
class TerrainCNN(nn.Module):
    def __init__(self):
        super(TerrainCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: 256 x 1 x 1
        )
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class MultimodalLandslideModel(nn.Module):
    def __init__(self):
        super(MultimodalLandslideModel, self).__init__()
        # Branch 1: Pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) 
        # Branch 2: Custom Terrain CNN
        self.terrain_cnn = TerrainCNN()
        # Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(2304, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
    def forward(self, rgb, terrain):
        f_rgb = self.resnet(rgb)
        f_rgb = f_rgb.view(f_rgb.size(0), -1)
        f_terrain = self.terrain_cnn(terrain)
        f_combined = torch.cat((f_rgb, f_terrain), dim=1)
        return self.fusion_head(f_combined)

# --- Caching Loaders ---
@st.cache_resource
def load_model():
    model_path = 'saved_models/multimodal_stage2.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalLandslideModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, device

@st.cache_data
def load_metadata():
    if os.path.exists('dataset_proposed/metadata.csv'):
        df = pd.read_csv('dataset_proposed/metadata.csv')
        df['npy_path'] = df['image_path'].str.replace('dataset_version_2', 'dataset_proposed').str.replace('.png', '.npy')
        df = df[df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
        return df
    return None

# --- Preprocessing ---
imagenet_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_sample(npy_path, device):
    data = np.load(npy_path).astype(np.float32) # Shape: (6, 224, 224)
    
    # Extract branches (C, H, W)
    rgb_t = torch.tensor(data[0:3, :, :])  # B4, B3, B2 (0-1 min-maxed)
    terrain_raw = data[3:6, :, :] # B8, DEM, Slope
    
    # Normalize Terrain (Min-Max per channel)
    terrain_norm = np.zeros_like(terrain_raw)
    for c in range(3):
        c_min, c_max = terrain_raw[c,:,:].min(), terrain_raw[c,:,:].max()
        if c_max > c_min:
            terrain_norm[c,:,:] = (terrain_raw[c,:,:] - c_min) / (c_max - c_min)
            
    terrain_t = torch.tensor(terrain_norm)
    
    # Apply ImageNet normalization to RGB
    rgb_t_norm = imagenet_transform(rgb_t)
    
    # Add batch dimension
    rgb_batch = rgb_t_norm.unsqueeze(0).to(device)
    terrain_batch = terrain_t.unsqueeze(0).to(device)
    
    # For visualization, we need (H, W, C)
    rgb_img = rgb_t.numpy().transpose(1, 2, 0)
    terrain_img = terrain_norm.transpose(1, 2, 0)
    
    return rgb_batch, terrain_batch, rgb_img, terrain_img

# --- UI Setup ---
st.title("🛰️ Multimodal Landslide Detection")
st.markdown("Test the **Multimodal Late-Fusion Model (ResNet50 + Terrain CNN)** on Sentinel-2 physical parameters.")

df = load_metadata()
model, device = load_model()

if model is None:
    st.error("Model weights not found! Ensure `saved_models/multimodal_stage2.pth` exists.")
    st.stop()
if df is None or len(df) == 0:
    st.error("Dataset not found! Ensure `dataset_proposed/metadata.csv` and patches exist.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=0.291, step=0.001)

st.sidebar.subheader("Select a Sample")
# Give samples friendly names
options = []
for idx, row in df.iterrows():
    label_str = "🛑 Landslide" if row['label'] == 1 else "✅ Safe"
    options.append(f"{label_str} | ID: {row['landslide_id']}")

selected_option = st.sidebar.selectbox("Choose a testing patch:", options)
selected_idx = options.index(selected_option)
selected_row = df.iloc[selected_idx]

st.sidebar.markdown("---")
st.sidebar.write("**Ground Truth Data:**")
st.sidebar.write(f"- Label: **{'Landslide (1)' if selected_row['label'] == 1 else 'No Landslide (0)'}**")
st.sidebar.write(f"- Latitude: {selected_row['latitude']:.4f}")
st.sidebar.write(f"- Longitude: {selected_row['longitude']:.4f}")
st.sidebar.write(f"- Cloud Cover: {selected_row['cloud_percentage']}%")

# --- Main App ---
st.write(f"### Analysis for `{selected_row['landslide_id']}`")

# Run Inference
rgb_tensor, terrain_tensor, rgb_img, terrain_img = process_sample(selected_row['npy_path'], device)

with torch.no_grad():
    output = model(rgb_tensor, terrain_tensor)
    prob = torch.sigmoid(output).item()

# Display Results
col_res1, col_res2 = st.columns(2)
is_landslide = prob >= threshold

with col_res1:
    st.metric(label="Model Probability", value=f"{prob*100:.1f}%")
    
with col_res2:
    if is_landslide:
        st.error("🚨 **PREDICTION: LANDSLIDE DETECTED**")
    else:
        st.success("✅ **PREDICTION: SAFE TERRAIN**")

st.markdown("---")
st.subheader("Data Visualization")
st.write("Model Inputs (RGB vs Physical Variables)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Sentinel-2 RGB values are 0-10000. A standard visual stretch is dividing by 3000.
    rgb_vis = np.clip(rgb_img / 3000.0, 0, 1)
    st.image(rgb_vis, caption="RGB (True Color)", use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(terrain_img[:,:,0], cmap='RdYlGn')
    ax.axis('off')
    st.pyplot(fig)
    st.caption("B08 (Near-Infrared)")
with col3:
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(terrain_img[:,:,1], cmap='terrain')
    ax.axis('off')
    st.pyplot(fig)
    st.caption("DEM (Elevation)")
with col4:
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(terrain_img[:,:,2], cmap='inferno')
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Slope Gradient")
