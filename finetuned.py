import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from glob import glob
import matplotlib.pyplot as plt
import cv2

# 1. Define parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# 2. Create the CheXNet model architecture
def create_chexnet(input_shape=(224, 224, 3), classes=14):
    base_model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=input_shape
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Create the model
model = create_chexnet()

# 3. Load pre-trained weights
weights_path = '/Users/nuthankishoremaddineni/Desktop/Radilogy_report_generation/weights.h5'
try:
    model.load_weights(weights_path)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")

# 4. Preprocess a sample image to check if our preprocessing works
def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to 3 channels by stacking
    img = np.stack([img, img, img], axis=-1)
    
    # Normalize
    img = img / 255.0
    
    return img

# Try preprocessing a sample image and print its shape
# Replace with a path to one of your X-ray images
sample_image_path = '/Users/nuthankishoremaddineni/Desktop/Radiology_Data/NLMCXR_png/CXR1_1_IM-0001-4001.png'
try:
    processed_img = preprocess_image(sample_image_path)
    print(f"Processed image shape: {processed_img.shape}")
    
    # Visualize to verify
    plt.imshow(processed_img)
    plt.title("Preprocessed X-ray Image")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"❌ Error preprocessing sample image: {e}")

# 5. Data preparation

def prepare_dataset(image_dir, label_mapping=None):
    """
    Prepare dataset from a directory of images.
    
    Args:
        image_dir: Directory containing images
        label_mapping: Function to extract labels from filenames or paths
        
    Returns:
        DataFrame with image paths and labels
    """
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    
    print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    # For demonstration, assign random labels (0 or 1) for binary classification
    # Replace this with your actual label extraction logic
    if label_mapping is None:
        # Default random labels for testing
        labels = np.random.randint(0, 2, size=len(image_paths))
    else:
        labels = [label_mapping(path) for path in image_paths]
    
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

# Replace with your image directory
image_dir = '/Users/nuthankishoremaddineni/Desktop/Radiology_Data/NLMCXR_png'

# Define your label mapping function
def get_label(image_path):
    # This is an example - implement your own logic based on your dataset
    if 'normal' in image_path.lower():
        return 0  # Normal
    else:
        return 1  # Abnormal
    
# Create dataset
try:
    df = prepare_dataset(image_dir, label_mapping=get_label)
    print(f"Dataset created with {len(df)} images")
    print(df.head())
except Exception as e:
    print(f"❌ Error creating dataset: {e}")

# Split into train and validation
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train: {len(train_df)}, Validation: {len(valid_df)}")

# 6. Custom data generator that properly handles grayscale to RGB conversion
from tensorflow.keras.utils import Sequence

# Replace the XrayDataGenerator class with this Sequence-based implementation
class XrayDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, img_size=224, shuffle=True, augment=False, num_classes=14):
        self.dataframe = dataframe.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        self.n = len(self.dataframe)
        self.indexes = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        # Define augmentation generator
        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.augmenter = None
    
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        batch_x = []
        batch_y = []
        
        for _, row in batch_df.iterrows():
            try:
                # Load and preprocess image
                img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read {row['image_path']}")
                    continue
                
                # Resize
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Convert to RGB by repeating the channel
                img = np.stack([img, img, img], axis=-1)
                
                # Normalize
                img = img / 255.0
                
                # Apply augmentation if needed
                if self.augment and self.augmenter:
                    img = self.augmenter.random_transform(img)
                
                batch_x.append(img)

                # Convert label to one-hot encoding (Multi-label classification)
                label = np.zeros(self.num_classes)  # Assuming 14 classes
                label[row['label']] = 1  # Assuming 'label' column contains indices (0-13)
                batch_y.append(label)
            except Exception as e:
                print(f"Error processing {row['image_path']}: {e}")
        
        if not batch_x:
            print("Warning: Empty batch created")
            return np.zeros((1, self.img_size, self.img_size, 3)), np.zeros((1, self.num_classes))
        
        return np.array(batch_x), np.array(batch_y)



# Create generators
train_generator = XrayDataGenerator(train_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=True)
valid_generator = XrayDataGenerator(valid_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=False)

# 7. Test data generator
def test_generator(generator):
    x_batch, y_batch = generator[0]
    print(f"Batch x shape: {x_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    
    # Plot a sample
    plt.figure(figsize=(10, 4))
    for i in range(min(3, len(x_batch))):
        plt.subplot(1, 3, i+1)
        plt.imshow(x_batch[i])
        plt.title(f"Label: {y_batch[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

try:
    test_generator(train_generator)
    print("✅ Data generator test successful!")
except Exception as e:
    print(f"❌ Error testing data generator: {e}")

# 8. Fine-tuning setup
# Freeze early layers to preserve learned features
for layer in model.layers[:-4]:  # Freeze all but the last 4 layers
    layer.trainable = False

# 9. Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',  # Adjust based on your task (binary vs multi-label)
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 10. Callbacks for training
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
x_batch, y_batch = train_generator[0]
print(f"Batch image shape: {x_batch.shape}")  # Expected: (batch_size, 224, 224, 3)
print(f"Batch label shape: {y_batch.shape}")  # Expected: (batch_size, 14)


# 11. Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

# 12. Evaluate the model
evaluation = model.evaluate(valid_generator)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Accuracy: {evaluation[1]}")
print(f"Validation AUC: {evaluation[2]}")

# 13. Save the fine-tuned model
model.save('chexnet_indiana_finetuned.h5')

# 14. Plot training history
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