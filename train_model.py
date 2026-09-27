import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 18

def prepare_split_dirs():
    """Create a clean, shuffled 80/20 train/val directory structure"""
    base_dir = 'dataset_split'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    for split in ['train', 'val']:
        for cls in ['healthy', 'disease']:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    for cls in ['healthy', 'disease']:
        src_dir = os.path.join('dataset', cls)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        random.seed(42)
        random.shuffle(files)

        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for f in train_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(base_dir, 'train', cls, f))
        for f in val_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(base_dir, 'val', cls, f))

    print("Clean split directories created in 'dataset_split/'")
    return base_dir

def build_balanced_cnn():
    model = Sequential([
        # Conv 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Conv 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Conv 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Dense Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    split_dir = prepare_split_dirs()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(split_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['healthy', 'disease'],
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(split_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['healthy', 'disease'],
        shuffle=False
    )

    # Calculate balanced class weights to prevent bias toward healthy class
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class indices: {train_gen.class_indices}")
    print(f"Computed Class Weights: {class_weight_dict}")

    model = build_balanced_cnn()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    print("\nStarting Balanced CNN Training on Eye Dataset...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[reduce_lr, early_stop]
    )

    model_path = 'eye_disease_model.h5'
    model.save(model_path)
    print(f"\nTraining Complete! Model saved to '{model_path}' successfully.")

if __name__ == '__main__':
    train()
