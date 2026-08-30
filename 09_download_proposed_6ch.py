import ee
import os
import requests
import numpy as np
import warnings
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tifffile
import io

warnings.filterwarnings('ignore')

# Initialize Earth Engine
print("Initializing Earth Engine...")
ee.Initialize(project='vision-based-landslide')

# ============================================================
# 6-Channel Proposed Model Data Download (Raw Numerical Values)
# Channels:
#   0: B4  (Red)          — Sentinel-2 visible
#   1: B3  (Green)        — Sentinel-2 visible
#   2: B2  (Blue)         — Sentinel-2 visible
#   3: B8  (NIR)          — Vegetation stress / land cover
#   4: DEM (Elevation)    — SRTM 30m absolute elevation
#   5: Slope              — Derived from DEM (degrees)
# shape: (6, 224, 224)
# ============================================================

def create_tasks():
    print("Loading samples from Baseline metadata (dataset_version_2)...")
    baseline_csv = "dataset_version_2/metadata.csv"
    if not os.path.exists(baseline_csv):
        print("Error: Baseline metadata.csv not found! Please run the baseline download first.")
        return []
        
    df = pd.read_csv(baseline_csv)
    tasks = []
    
    for _, row in df.iterrows():
        # Match the proposed dataset filenames
        filename = f"dataset_proposed/{'positive' if row['label'] == 1 else 'negative'}/{row['sample_id']}.npy"
        
        if not os.path.exists(filename):
            tasks.append({
                'sample_id': row['sample_id'],
                'landslide_id': row['landslide_id'],
                'lon': row['longitude'],
                'lat': row['latitude'],
                'label': row['label'],
                'acquisition_date': row['acquisition_date'],
                'filename': filename
            })
            
    print(f"Total new proposed patches to download: {len(tasks)}")
    return tasks


def download_patch_6ch(task):
    """
    Downloads a 6-channel patch and saves as .npy file:
      [B4, B3, B2, B8, DEM, Slope]  shape: (6, 224, 224)
    Uses GeoTIFF download to preserve exact physical values.
    """
    try:
        point = ee.Geometry.Point([task['lon'], task['lat']])
        region = point.buffer(1120).bounds()

        # Fetch specific date window from the baseline task to ensure identical seasonal conditions
        acq_date_str = task['acquisition_date']
        
        # We query the exact same image used in baseline or an image close to it
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(region)
              .filterDate('2025-01-01', '2025-10-31')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .sort('CLOUDY_PIXEL_PERCENTAGE')
              .first())

        if s2 is None:
            return None

        # --- DEM & Slope ---
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem)

        # Combine all 6 bands into one image
        # S2: B4, B3, B2, B8  (range ~0-10000 Surface Reflectance)
        # DEM: elevation in meters
        # Slope: 0-90 degrees
        combined = (s2.select(['B4', 'B3', 'B2', 'B8'])
                      .addBands(dem.rename('DEM'))
                      .addBands(slope.rename('Slope')))

        # Download raw numerical values as a multi-band GeoTIFF
        url = combined.getDownloadURL({
            'region': region,
            'dimensions': '224x224',
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return None

        # Read GeoTIFF as numpy array (shape: H, W, Channels)
        raw_array = tifffile.imread(io.BytesIO(response.content))
        
        # TIFF might return (224, 224, 6) or (6, 224, 224). 
        # Usually rasterio/tifffile from Earth Engine returns (224, 224, 6) if multi-band
        if raw_array.shape[-1] == 6:
            # Convert to (6, 224, 224) for PyTorch
            patch = np.transpose(raw_array, (2, 0, 1))
        elif raw_array.shape[0] == 6:
            patch = raw_array
        else:
            return None

        # Convert to float32
        patch = patch.astype(np.float32)

        # Save the numerical array
        np.save(task['filename'], patch)

        # Retrieve metadata for the CSV
        info = s2.getInfo()
        props = info.get('properties', {})
        cloud_pct = props.get('CLOUDY_PIXEL_PERCENTAGE', 0)
        time_start = props.get('system:time_start', 0)
        actual_acq_date = datetime.fromtimestamp(time_start / 1000.0).strftime('%Y-%m-%d') if time_start > 0 else 'Unknown'

        row = {
            'sample_id': task['sample_id'],
            'latitude': task['lat'],
            'longitude': task['lon'],
            'landslide_id': task['landslide_id'],
            'label': task['label'],
            'image_path': task['filename'],
            'satellite': 'Sentinel-2',
            'acquisition_date': actual_acq_date,
            'cloud_percentage': round(cloud_pct, 2),
            'channels': 'B4,B3,B2,B8,DEM,Slope',
            'resolution': '10m'
        }
        return row

    except Exception as e:
        return None


if __name__ == '__main__':
    os.makedirs('dataset_proposed/positive', exist_ok=True)
    os.makedirs('dataset_proposed/negative', exist_ok=True)

    tasks = create_tasks()
    if not tasks:
        print("All patches already downloaded!")
        exit()

    csv_path = 'dataset_proposed/metadata.csv'
    existing_data = []
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path).to_dict('records')

    results = []
    print("Starting 6-channel raw numerical patch download...")

    # Using thread pool (fewer workers as it processes full GeoTIFFs)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for row in tqdm(executor.map(download_patch_6ch, tasks), total=len(tasks), desc="Downloading 6ch patches"):
            if row:
                results.append(row)

    all_data = existing_data + results
    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=['sample_id'], inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"\nDownloaded {len(results)} new patches.")
        print(f"Total in metadata: {len(df)}")
        print(f"Saved to: dataset_proposed/")
        print(f"Each patch shape: (6, 224, 224) raw values saved as .npy")
    else:
        print("\nNo new patches were downloaded.")
