import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

# Enable Mixed Precision for Faster Training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Dataset Paths
train_dir = "C:\\Users\\User\\project\\images\\train"
validation_dir = "C:\\Users\\User\\project\\images\\test"

# Optimized Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(validation_generator.classes),
    y=validation_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Load Pretrained MobileNetV2 and Freeze More Layers Initially
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
for layer in base_model.layers[:-10]:  # Unfreeze only last 10 layers
    layer.trainable = False

# Optimized Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(7, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
model.compile(
    optimizer=AdamW(learning_rate=0.0005, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stopping]
)

# Save Model
model.save("optimized_face_expression_model.h5")
print("Optimized Model Trained and Saved")
