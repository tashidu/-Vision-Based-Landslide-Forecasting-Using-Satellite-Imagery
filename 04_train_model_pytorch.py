import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score

# ── 1. Setup CUDA / Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"Using Device   : {device}")
if device.type == 'cuda':
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"CUDA Version   : {torch.version.cuda}")
else:
    print("WARNING: CUDA GPU not detected. Training will run on CPU.")
print("=" * 60)

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ── 2. Load Dataset & Geographic Split ──────────────────────────────────────
metadata_path = 'dataset_version_2/metadata.csv'
if not os.path.exists(metadata_path):
    metadata_path = 'metadata.csv'

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please check your dataset folder.")

df = pd.read_csv(metadata_path)
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"\nTotal valid images loaded: {len(df)}")

# Enforce strict geographic split using GroupShuffleSplit on landslide_id
gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['landslide_id']))
df_train = df.iloc[train_idx].copy().reset_index(drop=True)
df_temp  = df.iloc[temp_idx].copy().reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['landslide_id']))
df_val  = df_temp.iloc[val_idx].copy().reset_index(drop=True)
df_test = df_temp.iloc[test_idx].copy().reset_index(drop=True)

print(f"Split sizes -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
overlap = len(set(df_train['landslide_id']) & set(df_val['landslide_id']))
print(f"Geographic Leakage Check (Train/Val overlap): {overlap} (Must be 0)")

# Calculate class distribution & positive weight for BCEWithLogitsLoss
neg_count = (df_train['label'] == 0).sum()
pos_count = (df_train['label'] == 1).sum()
pos_weight = torch.tensor([neg_count / float(pos_count)], dtype=torch.float32).to(device)
print(f"Class counts -> Negative: {neg_count} | Positive: {pos_count} | pos_weight: {pos_weight.item():.3f}")

# ── 3. Custom PyTorch Dataset & Transforms ─────────────────────────────────
class LandslideDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = float(self.df.iloc[idx]['label'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = LandslideDataset(df_train, transform=train_transform)
val_dataset   = LandslideDataset(df_val,   transform=val_test_transform)
test_dataset  = LandslideDataset(df_test,  transform=val_test_transform)

# Multi-threaded dataloaders
num_workers = 0 if os.name == 'nt' else 2
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=(device.type=='cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

# ── 4. Model Architectures ──────────────────────────────────────────────────

class ImprovedBaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1 - 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # Block 2 - 64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # Block 3 - 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # Global Average Pooling & Classifier Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet50Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_trainable(self, trainable=True, unfreeze_layers_from='layer4'):
        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Always keep fc head trainable
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            # Freeze early layers, unfreeze specific upper layers
            unfreeze = False
            for name, child in self.backbone.named_children():
                if name == unfreeze_layers_from:
                    unfreeze = True
                if unfreeze or name == 'fc':
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False

# ── 5. Training Loop Helper ────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, save_filename, patience=5):
    model.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    print(f"\nBeginning training on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_preds_arr = np.array(val_preds).flatten()
        val_targets_arr = np.array(val_targets).flatten()
        
        val_acc = accuracy_score(val_targets_arr, (val_preds_arr > 0.5).astype(int))
        try:
            val_auc = roc_auc_score(val_targets_arr, val_preds_arr)
        except Exception:
            val_auc = 0.5

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Checkpoint & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_filename)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"--> Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} mins. Saved best weights to '{save_filename}'")
    # Load best model weights
    model.load_state_dict(torch.load(save_filename, weights_only=True))
    return model, history

# ── 6. Evaluation Helper ───────────────────────────────────────────────────
def evaluate_model(model, name, test_loader):
    model.eval()
    model.to(device)
    test_preds, test_targets = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs)
            test_targets.extend(labels.numpy())

    y_pred_probs = np.array(test_preds).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    y_true = np.array(test_targets).flatten()

    print(f"\n==========================================")
    print(f"--- Evaluation: {name} ---")
    print(f"==========================================")
    print(classification_report(y_true, y_pred, target_names=['No Landslide', 'Landslide']))
    
    auc = roc_auc_score(y_true, y_pred_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Landslide', 'Landslide'],
                yticklabels=['No Landslide', 'Landslide'])
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    filename = f'confusion_matrix_{name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to '{filename}'")

# ── 7. Run Model 1: Improved Baseline CNN ──────────────────────────────────
print("\n" + "="*60)
print("=== MODEL 1: PyTorch Improved Baseline CNN ===")
print("="*60)

baseline_model = ImprovedBaselineCNN()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

baseline_model, history_baseline = train_model(
    baseline_model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs=20, save_filename='baseline_cnn_pytorch.pth', patience=5
)
evaluate_model(baseline_model, "PyTorch_Improved_Baseline_CNN", test_loader)

# ── 8. Run Model 2: ResNet50 Transfer Learning ──────────────────────────────
print("\n" + "="*60)
print("=== MODEL 2: PyTorch ResNet50 Transfer Learning ===")
print("="*60)

resnet_model = ResNet50Transfer()

# Stage 1: Train Head Only
print("\n--- Stage 1: Training Classifier Head (Frozen Backbone) ---")
resnet_model.set_backbone_trainable(trainable=False)
optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=1e-4)
scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', factor=0.5, patience=3)

resnet_model, history_resnet1 = train_model(
    resnet_model, train_loader, val_loader, criterion, optimizer_stage1, scheduler_stage1,
    epochs=15, save_filename='resnet50_pytorch_stage1.pth', patience=4
)

# Stage 2: Fine-tune Upper ResNet Layers (layer4)
print("\n--- Stage 2: Fine-Tuning Upper Backbone (layer4) ---")
resnet_model.set_backbone_trainable(trainable=True, unfreeze_layers_from='layer4')
optimizer_stage2 = optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=1e-5)
scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode='min', factor=0.5, patience=3)

resnet_model, history_resnet2 = train_model(
    resnet_model, train_loader, val_loader, criterion, optimizer_stage2, scheduler_stage2,
    epochs=10, save_filename='resnet50_pytorch_final.pth', patience=4
)

evaluate_model(resnet_model, "PyTorch_ResNet50_FineTuned", test_loader)

print("\n" + "="*60)
print("ALL PYTORCH MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
print("="*60)
