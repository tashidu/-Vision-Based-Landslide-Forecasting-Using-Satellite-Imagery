# Vision-Based Landslide Forecasting: Workflow & Baseline Model Documentation

This document explains the data pipeline and the **Baseline CNN** training process. You can use these details directly in your assignment report.

## 1. Data Structure & Preprocessing

We are using satellite imagery obtained from Google Earth Engine, centered around the 2025 Ditwah landslide events.

### Data Storage Format
The images are stored in a single folder, and a `metadata.csv` file maps each image to its label and geographic group. This is much cleaner than relying purely on folder structures, as it allows us to do complex geographic splitting.

```text
dataset_version_2/
├── metadata.csv
├── positive/
│   ├── pos_0.png
│   ├── pos_1.png
│   └── ...
└── negative/
    ├── neg_0.png
    ├── neg_1.png
    └── ...
```

### Metadata Structure (`metadata.csv`)
*   `image_path`: Relative path to the `.png` image (e.g., `dataset_version_2/positive/pos_0.png`).
*   `label`: `1` for Landslide (Positive), `0` for Non-Landslide (Negative).
*   `group_id` (currently `landslide_id`): A unique ID representing the geographic cluster/region. It links a positive landslide with its nearby negative sample pair to ensure they stay in the same split.

## 2. Preventing Data Leakage (Geographic Splitting)

The assignment strictly penalizes **Data Leakage**. If we use standard random splitting (like `train_test_split`), a positive landslide patch and a negative patch from the exact same mountain could end up in different sets, artificially boosting accuracy.

**Our Solution:** We use `GroupShuffleSplit` grouped by our geographic identifier (`landslide_id`).
This ensures that if a specific landslide area is placed in the Test set, its corresponding negative samples are also placed in the Test set. The model never sees that geographic region during training.

```python
# 70% Train, 30% Temp
gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['landslide_id']))

df_train = df.iloc[train_idx].copy()
df_temp = df.iloc[temp_idx].copy()

# Split the Temp 30% into Val (15%) and Test (15%)
gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['landslide_id']))

df_val = df_temp.iloc[val_idx].copy()
df_test = df_temp.iloc[test_idx].copy()
```

## 3. PyTorch Custom Dataset

Since our data is managed by a CSV, we wrote a custom PyTorch `Dataset`.

```python
class LandslideDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = float(self.dataframe.iloc[idx]['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)
```
**Data Augmentation:** Data augmentation was applied to the training set to improve model generalization and reduce the risk of overfitting. Transformations include `RandomHorizontalFlip()`, `RandomVerticalFlip()`, and `RandomRotation(30)`.

## 4. Baseline CNN Architecture

The baseline CNN learns directly from pixel-level patterns in the pre-event RGB satellite imagery without explicitly incorporating terrain-based domain knowledge. During training, the convolutional layers progressively learn increasingly complex spatial features. Early layers may learn low-level patterns such as edges, intensity variations, and local textures, while deeper layers can learn combinations of these patterns that are useful for distinguishing future landslide locations from comparison areas.

The model does not explicitly understand concepts such as "forest," "bare soil," or "landslide scar." Instead, it learns statistical relationships between visual patterns in the input images and the provided class labels. Therefore, the baseline investigates whether RGB satellite imagery alone contains sufficient visual information to distinguish locations that subsequently experienced landslides from locations that did not.

```python
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1) # Binary classification output
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

## 5. Training and Saving the Model

We use **Binary Cross Entropy Loss** (`BCEWithLogitsLoss`) with positive class weighting to handle any imbalances. We also use a learning rate scheduler (`ReduceLROnPlateau`) to automatically lower the learning rate if validation loss stops improving.

Once training completes, the best performing model is saved as a PyTorch state dictionary (`.pth` file).

```python
# Train the model
baseline_model = train_model(baseline_model, criterion, optimizer, scheduler, num_epochs=20, patience=5)

# Save the trained weights to disk
os.makedirs('saved_models', exist_ok=True)
torch.save(baseline_model.state_dict(), 'saved_models/baseline_cnn_final.pth')
```

This saved `.pth` file can later be loaded by our Streamlit UI to make real-time predictions without needing to retrain the model.
