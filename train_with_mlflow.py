# File: train_with_mlflow.py
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
import numpy as np
import os
import json

# --- Konfigurasi MLflow ---
mlflow.set_tracking_uri("http://localhost:5000") # Sesuaikan jika menggunakan server remote
mlflow.set_experiment("Flower_Classification_ITERA")

DATASET_DIR = 'Dataset Bunga' # Pastikan folder ini ada
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50 
LEARNING_RATE = 0.001

def train():
    # Mulai Run MLflow
    with mlflow.start_run():
        # Log Parameter
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("model_type", "Custom_CNN_Nano")

        # --- Data Preparation (Sama seperti sebelumnya) ---
        train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=30, width_shift_range=0.25,
            height_shift_range=0.25, shear_range=0.2, zoom_range=0.3,
            horizontal_flip=True, fill_mode='nearest', validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='sparse', subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory( # Gunakan train_datagen untuk split yang konsisten atau buat instance baru
            DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='sparse', subset='validation'
        )

        NUM_CLASSES = len(train_generator.class_indices)
        
        # Simpan class names untuk API
        with open('class_names.json', 'w') as f:
            json.dump(list(train_generator.class_indices.keys()), f)
        mlflow.log_artifact('class_names.json')

        # --- Arsitektur Model ---
        model = Sequential([
            Input(shape=(*IMG_SIZE, 3)),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            GlobalMaxPooling2D(),
            Dense(128, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # --- Callbacks ---
        # MLflow Autologging (Akan otomatis mencatat metrics per epoch)
        mlflow.tensorflow.autolog()
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Hitung Class Weights
        training_classes = train_generator.classes
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(training_classes), y=training_classes
        )
        class_weights_dict = dict(enumerate(class_weights))

        # --- Training ---
        print("Mulai Training...")
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            callbacks=[reduce_lr, early_stopping],
            class_weight=class_weights_dict
        )

        # Simpan Model Lokal (Untuk disalin ke folder production nanti)
        model.save('models/flower_classifier_new.keras')
        print("Training Selesai.")

if __name__ == "__main__":
    train()