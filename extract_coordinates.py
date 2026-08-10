import geopandas as gpd

# Path to the shapefile
shapefile_path = r"D:\3rd year\4th year\vision_based\Landslides\SHP\Lanslides_Ditwah_2025.shp"

def main():
    print(f"Loading shapefile from: {shapefile_path}")
    
    # Read the shapefile
    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    # Print basic information
    print("\n--- Basic Information ---")
    print(f"Total number of landslide records (polygons): {len(gdf)}")
    print(f"Coordinate Reference System (CRS): {gdf.crs}")
    print("\n--- First 5 rows of data ---")
    print(gdf.head())

    # Calculate centroids (the middle point of each landslide polygon)
    # This is important because to download a satellite image, we usually need a center point.
    print("\n--- Calculating Centroids (Center Points) ---")
    
    # We should ensure the CRS is geographical (EPSG:4326) to get proper Lat/Lon coordinates
    if gdf.crs != "EPSG:4326":
        print("Converting to EPSG:4326 (Latitude/Longitude)...")
        gdf = gdf.to_crs("EPSG:4326")

    # Get the centroid of each polygon
    gdf['centroid'] = gdf.geometry.centroid
    
    # Extract Longitude (x) and Latitude (y) into separate columns
    gdf['longitude'] = gdf['centroid'].x
    gdf['latitude'] = gdf['centroid'].y

    # Show the first few coordinates
    print("First 5 Landslide Center Coordinates:")
    print(gdf[['longitude', 'latitude']].head())

    # You can save this cleaned data to a CSV file if you want to use it later without geopandas
    # gdf[['longitude', 'latitude']].to_csv("landslide_coordinates.csv", index=False)
    # print("\nCoordinates saved to landslide_coordinates.csv")

if __name__ == "__main__":
    main()
