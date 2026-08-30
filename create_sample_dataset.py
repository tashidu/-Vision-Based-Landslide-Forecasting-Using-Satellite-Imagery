import os
import shutil
import pandas as pd

# 1. Read original metadata
original_meta_path = 'dataset_proposed/metadata.csv'
df = pd.read_csv(original_meta_path)

# 2. Filter for files that exist as .npy
df['npy_path'] = df['image_path'].str.replace('dataset_version_2', 'dataset_proposed').str.replace('.png', '.npy')
df = df[df['npy_path'].apply(os.path.exists)]

# 3. Select 10 landslides and 10 safe
df_landslide = df[df['label'] == 1].head(10)
df_safe = df[df['label'] == 0].head(10)
df_sample = pd.concat([df_landslide, df_safe]).reset_index(drop=True)

# 4. Create sample directory
sample_dir = 'dataset_sample'
os.makedirs(sample_dir, exist_ok=True)

# 5. Copy files and update paths
new_rows = []
for _, row in df_sample.iterrows():
    old_npy = row['npy_path']
    filename = os.path.basename(old_npy)
    new_npy = os.path.join(sample_dir, filename)
    
    # Copy file
    shutil.copy(old_npy, new_npy)
    
    # Update row (keep image_path for consistency, or adjust as needed)
    new_row = row.copy()
    # The app.py replaces 'dataset_version_2' with the dataset name.
    # So we'll rewrite image_path to point to dataset_version_2/... so app.py can replace it
    # Wait, in the Space, we'll just modify app.py to replace with 'dataset_sample'
    new_rows.append(new_row)

# 6. Save new metadata
df_final = pd.DataFrame(new_rows)
df_final = df_final.drop(columns=['npy_path']) # Clean up
df_final.to_csv(os.path.join(sample_dir, 'metadata.csv'), index=False)

print("Sample dataset created successfully!")
