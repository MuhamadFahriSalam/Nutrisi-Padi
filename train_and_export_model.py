# train_and_export_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import json

# Konfigurasi
dataset_dir = 'rice_plant_lacks_nutrients' # Ganti dengan path dataset Anda
img_size = (224, 224)
batch_size = 16

# ImageDataGenerator untuk preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Membangun model CNN
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# Kompilasi & Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# Simpan model
model.save("rice_leaf_diseases.keras") # Pastikan ekstensi .keras digunakan untuk kompatibilitas nama file seperti yang diinginkan

# Simpan mapping label
with open("label_map.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("Model dan label map berhasil disimpan.")
