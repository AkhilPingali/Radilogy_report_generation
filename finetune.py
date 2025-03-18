import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from glob import glob

# 1. Load the pre-trained CheXNet model
base_model = load_model('/Users/nuthankishoremaddineni/Desktop/Radilogy_report_generation/brucechou1983_CheXNet_Keras_0.3.0_weights (1).h5')

# 2. Define parameters
IMG_SIZE = 224  # CheXNet typically uses 224x224 images
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# 3. Data preparation
# Assuming your X-ray images are organized in a directory structure
image_paths = glob('path/to/indiana/dataset/*.png')  # Adjust path and extension as needed

# Create a DataFrame with image paths and labels
# You'll need to adapt this based on how your Indiana dataset is labeled
def create_dataframe(image_paths):
    data = {
        'image_path': image_paths,
        'label': [extract_label_from_path(path) for path in image_paths]  # You'll need to implement this function
    }
    return pd.DataFrame(data)

# This is a placeholder function - you'll need to implement based on your data structure
def extract_label_from_path(path):
    # Example: if your filenames or directories contain label information
    # For binary classification (e.g., normal vs pneumonia)
    if 'normal' in path.lower():
        return 0
    else:
        return 1

df = create_dataframe(image_paths)

# Split into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 4. Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Custom generator functions to load and preprocess images
def generate_data(dataframe, batch_size, augment=False):
    datagen = train_datagen if augment else valid_datagen
    i = 0
    while True:
        batch_paths = dataframe['image_path'][i:i+batch_size]
        batch_labels = dataframe['label'][i:i+batch_size]
        
        batch_images = []
        for path in batch_paths:
            img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
            img = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img)
        
        batch_images = np.array(batch_images)
        batch_images = datagen.flow(batch_images, batch_size=batch_size).next()[0]
        
        i += batch_size
        if i >= len(dataframe):
            i = 0
            
        yield batch_images, np.array(batch_labels)

train_generator = generate_data(train_df, BATCH_SIZE, augment=True)
valid_generator = generate_data(valid_df, BATCH_SIZE, augment=False)

# 5. Fine-tuning setup
# Freeze early layers to preserve learned features
for layer in base_model.layers[:-4]:  # Freeze all but the last 4 layers
    layer.trainable = False

# 6. Compile the model
base_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',  # Adjust based on your task (binary vs multi-label)
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 7. Callbacks for training
checkpoint = ModelCheckpoint(
    'chexnet_finetuned.h5',
    monitor='val_auc',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=5,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.2,
    patience=3,
    verbose=1,
    mode='max',
    min_lr=1e-6
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# 8. Train the model
history = base_model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid_generator,
    validation_steps=len(valid_df) // BATCH_SIZE,
    callbacks=callbacks
)

# 9. Evaluate the model
evaluation = base_model.evaluate(valid_generator, steps=len(valid_df) // BATCH_SIZE)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Accuracy: {evaluation[1]}")
print(f"Validation AUC: {evaluation[2]}")

# 10. Save the fine-tuned model
base_model.save('chexnet_indiana_finetuned.h5')

# 11. Optional: Plot training history
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)