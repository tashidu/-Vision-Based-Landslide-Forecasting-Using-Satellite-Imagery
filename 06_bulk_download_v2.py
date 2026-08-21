import ee
import os
import requests
import random
import geopandas as gpd
import warnings
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Initialize Earth Engine
print("Initializing Earth Engine...")
ee.Initialize(project='vision-based-landslide')

def create_tasks():
    print("Loading coordinates from shapefile...")
    shapefile_path = r"D:\3rd year\4th year\vision_based\Landslides\SHP\Lanslides_Ditwah_2025.shp"
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        
    gdf['centroid'] = gdf.geometry.centroid
    
    random.seed(42) # For reproducible negative samples
    tasks = []
    
    # Positive samples
    for idx, pt in enumerate(gdf['centroid']):
        filename = f'dataset_version_2/positive/landslide_{idx}.png'
        if not os.path.exists(filename):
            tasks.append({
                'landslide_id': f'LS_{idx:04d}',
                'lon': pt.x,
                'lat': pt.y,
                'label': 1,
                'filename': filename
            })
            
    # Negative samples
    for idx, pt in enumerate(gdf['centroid']):
        filename = f'dataset_version_2/negative/non_landslide_{idx}.png'
        lat_shift = random.choice([1, -1]) * random.uniform(0.02, 0.04)
        lon_shift = random.choice([1, -1]) * random.uniform(0.02, 0.04)
        if not os.path.exists(filename):
            tasks.append({
                'landslide_id': f'LS_{idx:04d}',
                'lon': pt.x + lon_shift,
                'lat': pt.y + lat_shift,
                'label': 0,
                'filename': filename
            })
            
    print(f"Total new images to download: {len(tasks)}")
    return tasks

def download_patch(task):
    try:
        point = ee.Geometry.Point([task['lon'], task['lat']])
        region = point.buffer(1120).bounds()
        
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate('2025-01-01', '2025-10-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
            
        if collection.size().getInfo() == 0:
            return None
            
        image = collection.first()
        info = image.getInfo()
        props = info.get('properties', {})
        
        cloud_pct = props.get('CLOUDY_PIXEL_PERCENTAGE', 0)
        satellite = props.get('SPACECRAFT_NAME', 'Sentinel-2')
        time_start = props.get('system:time_start', 0)
        
        if time_start > 0:
            acq_date = datetime.fromtimestamp(time_start / 1000.0).strftime('%Y-%m-%d')
        else:
            acq_date = 'Unknown'
            
        vis_image = image.select(['B4', 'B3', 'B2']).visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])
        url = vis_image.getThumbURL({
            'region': region,
            'dimensions': '224x224',
            'format': 'png'
        })
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(task['filename'], 'wb') as f:
                f.write(response.content)
            
            row = {
                'sample_id': os.path.basename(task['filename']).split('.')[0],
                'latitude': task['lat'],
                'longitude': task['lon'],
                'landslide_id': task['landslide_id'],
                'label': task['label'],
                'image_path': task['filename'],
                'satellite': satellite,
                'acquisition_date': acq_date,
                'cloud_percentage': round(cloud_pct, 2),
                'resolution': '10m'
            }
            return row
    except Exception as e:
        return None
    return None

if __name__ == '__main__':
    os.makedirs('dataset_version_2/positive', exist_ok=True)
    os.makedirs('dataset_version_2/negative', exist_ok=True)
    
    tasks = create_tasks()
    if not tasks:
        print("All images are already downloaded!")
        exit()
        
    csv_path = 'dataset_version_2/metadata.csv'
    existing_data = []
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path).to_dict('records')
        
    results = []
    print("Starting high-speed bulk download (V2)...")
    
    # Use 15 workers for very fast download speeds
    with ThreadPoolExecutor(max_workers=15) as executor:
        for row in tqdm(executor.map(download_patch, tasks), total=len(tasks), desc="Downloading"):
            if row:
                results.append(row)
                
    # Combine and save
    all_data = existing_data + results
    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=['sample_id'], inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSuccessfully downloaded and generated metadata for {len(results)} new images.")
        print(f"Total images in dataset metadata: {len(df)}")
    else:
        print("\nNo new images were downloaded.")
