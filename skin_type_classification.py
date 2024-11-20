import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error

# Load train and valid data
train_data = pd.read_csv('/Users/daksha/Desktop/kavs/train.txt', delimiter='\t', header=None, names=['image_path', 'label'])
valid_data = pd.read_csv('/Users/daksha/Desktop/kavs/valid.txt', delimiter='\t', header=None, names=['image_path', 'label'])

# Define image size and batch size
img_size = (128, 128)
batch_size = 32

# Convert labels to numeric for internal use, but keep original labels for the generator
label_to_num = {'dry': 0, 'normal': 1, 'oily': 2}

# Add a numeric label column for training
train_data['numeric_label'] = train_data['label'].map(label_to_num)
valid_data['numeric_label'] = valid_data['label'].map(label_to_num)

# Data generators for training and validation with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

# Create the train generator using the original string labels
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',  # Use original label for categorical
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the validation generator using the original string labels
valid_generator = valid_datagen.flow_from_dataframe(
    valid_data,
    x_col='image_path',
    y_col='label',  # Use original label for categorical
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build a deeper CNN model
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),  # New layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Increased number of units
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: dry, normal, oily
])

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model without workers, use_multiprocessing, or max_queue_size
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    validation_data=valid_generator,
    validation_steps=len(valid_data) // batch_size,
    epochs=20,  # Increased number of epochs
    verbose=0
)

# Evaluate on validation data without any output
val_loss, val_acc = model.evaluate(valid_generator, steps=len(valid_data) // batch_size, verbose=0)

# Save the trained model without print statements
model.save('skin_type_model.keras')
