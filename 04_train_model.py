import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                     GlobalAveragePooling2D, BatchNormalization)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ── Load dataset ──────────────────────────────────────────────────────────────
metadata_path = 'dataset_version_2/metadata.csv'
df = pd.read_csv(metadata_path)
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Total valid images: {len(df)}")

# ── Geographic split ──────────────────────────────────────────────────────────
gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['landslide_id']))
df_train = df.iloc[train_idx].copy()
df_temp  = df.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['landslide_id']))
df_val  = df_temp.iloc[val_idx].copy()
df_test = df_temp.iloc[test_idx].copy()

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
print("Train/Val overlap:", len(set(df_train['landslide_id']) & set(df_val['landslide_id'])))

# ── Key settings (reduced for CPU memory) ────────────────────────────────────
IMG_SIZE   = (128, 128)   # 224->128: memory 3x adu wenawa
BATCH_SIZE = 16           # 32->16:  RAM crash nawathennawa

df_train['label_str'] = df_train['label'].astype(str)
df_val['label_str']   = df_val['label'].astype(str)
df_test['label_str']  = df_test['label'].astype(str)

# ── Data generators ───────────────────────────────────────────────────────────
cnn_train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)
cnn_val_test_datagen = ImageDataGenerator(rescale=1./255)

resnet_train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.15,
    fill_mode='nearest'
)
resnet_val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def make_gen(datagen_train, datagen_val):
    classes = ['0', '1']
    train = datagen_train.flow_from_dataframe(
        df_train, x_col='image_path', y_col='label_str', classes=classes,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True)
    val = datagen_val.flow_from_dataframe(
        df_val, x_col='image_path', y_col='label_str', classes=classes,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)
    test = datagen_val.flow_from_dataframe(
        df_test, x_col='image_path', y_col='label_str', classes=classes,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)
    return train, val, test

cnn_train_gen,    cnn_val_gen,    cnn_test_gen    = make_gen(cnn_train_datagen,    cnn_val_test_datagen)
resnet_train_gen, resnet_val_gen, resnet_test_gen = make_gen(resnet_train_datagen, resnet_val_test_datagen)

# ── Class weights ─────────────────────────────────────────────────────────────
neg, pos = np.bincount(df_train['label'])
total = neg + pos
class_weight = {0: (1/neg)*(total/2.0), 1: (1/pos)*(total/2.0)}
print(f"Class weights -> 0: {class_weight[0]:.3f} | 1: {class_weight[1]:.3f}")

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1)
]

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1: IMPROVED BASELINE CNN (light version for CPU)
# vs original: BatchNorm added, GlobalAvgPool replaces Flatten,
#              double conv per block, 3 blocks (32->64->128)
# ═════════════════════════════════════════════════════════════════════════════
print("\n==================================")
print("--- Training Improved Baseline CNN ---")
print("==================================")

baseline_model = Sequential([
    # Block 1 - 32 filters
    Conv2D(32, (3,3), padding='same', activation='relu',
           input_shape=(128, 128, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 2 - 64 filters
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 3 - 128 filters
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Classifier head
    GlobalAveragePooling2D(),       # Flatten() replace - memory godak adu
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='Improved_Baseline_CNN')

baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)
baseline_model.summary()

history_baseline = baseline_model.fit(
    cnn_train_gen,
    validation_data=cnn_val_gen,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks
)

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2: ResNet50 Transfer Learning
# ═════════════════════════════════════════════════════════════════════════════
print("\n==================================")
print("--- Training ResNet50 (Transfer Learning) ---")
print("==================================")

base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False,
    input_shape=(128, 128, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

resnet_model = Model(inputs=base_model.input, outputs=output,
                     name='ResNet50_Transfer')
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Stage 1: frozen base
history_resnet = resnet_model.fit(
    resnet_train_gen,
    validation_data=resnet_val_gen,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks
)

# Stage 2: fine-tune last 20 layers
print("\n--- Fine-tuning ResNet50 last 20 layers ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)
resnet_model.fit(
    resnet_train_gen,
    validation_data=resnet_val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks
)

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, name, test_gen):
    print(f"\n{'='*40}\n--- Evaluating: {name} ---\n{'='*40}")
    preds  = model.predict(test_gen)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred,
          target_names=['No Landslide', 'Landslide']))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Landslide','Landslide'],
                yticklabels=['No Landslide','Landslide'])
    plt.title(f'Confusion Matrix: {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ","_")}.png', dpi=150)
    plt.close()
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true, preds.flatten()):.4f}")
    except Exception as e:
        print(f"AUC error: {e}")

evaluate_model(baseline_model, "Improved_Baseline_CNN", cnn_test_gen)
evaluate_model(resnet_model,   "ResNet50_FineTuned",    resnet_test_gen)
print("\nDone! Check the saved confusion matrix images.")
