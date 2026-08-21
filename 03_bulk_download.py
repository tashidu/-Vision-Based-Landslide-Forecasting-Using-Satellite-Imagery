import ee
import os
import requests
import random
import geopandas as gpd
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Initialize Earth Engine
ee.Initialize(project='vision-based-landslide')

def generate_dataset(shapefile_path):
    print("Loading coordinates from shapefile...")
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    gdf['centroid'] = gdf.geometry.centroid
    
    positive_coords = [(pt.x, pt.y) for pt in gdf['centroid']]
    print(f"Total positive coordinates: {len(positive_coords)}")
    
    # Generate negative coords
    random.seed(42) # For reproducibility (always generates same negative points)
    negative_coords = []
    for lon, lat in positive_coords:
        lat_shift = random.choice([1, -1]) * random.uniform(0.02, 0.04)
        lon_shift = random.choice([1, -1]) * random.uniform(0.02, 0.04)
        negative_coords.append((lon + lon_shift, lat + lat_shift))
        
    return positive_coords, negative_coords

def download_patch(args):
    lon, lat, filename = args
    
    # Skip if already downloaded
    if os.path.exists(filename):
        return True
        
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(1120).bounds()
        
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate('2025-12-05', '2026-04-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
            
        if collection.size().getInfo() == 0:
            return False
            
        image = collection.first().select(['B4', 'B3', 'B2'])
        vis_image = image.visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])
        
        url = vis_image.getThumbURL({
            'region': region,
            'dimensions': '224x224',
            'format': 'png'
        })
        
        response = requests.get(url, timeout=15)
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    shapefile = r"D:\3rd year\4th year\vision_based\Landslides\SHP\Lanslides_Ditwah_2025.shp"
    pos_coords, neg_coords = generate_dataset(shapefile)
    
    os.makedirs('dataset/positive', exist_ok=True)
    os.makedirs('dataset/negative', exist_ok=True)
    
    # Create a list of all download tasks
    tasks = []
    for i, (lon, lat) in enumerate(pos_coords):
        tasks.append((lon, lat, f'dataset/positive/landslide_{i}.png'))
        
    for i, (lon, lat) in enumerate(neg_coords):
        tasks.append((lon, lat, f'dataset/negative/non_landslide_{i}.png'))
        
    print(f"Starting download of {len(tasks)} images (This will take some time)...")
    
    # Use ThreadPoolExecutor to download 10 images at the same time (faster)
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in tqdm(executor.map(download_patch, tasks), total=len(tasks), desc="Downloading"):
            if result:
                success_count += 1
                
    print(f"\nDownload complete! Successfully downloaded {success_count}/{len(tasks)} images.")
