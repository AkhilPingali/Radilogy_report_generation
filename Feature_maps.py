import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from PIL import Image
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Model

# === 1. Build CheXNet Feature Extractor ===
def build_chexnet_feature_extractor(weights_path):
    base_model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    base_model.load_weights(weights_path, by_name=True)
    model = Model(inputs=base_model.input, outputs=base_model.output)  # (7, 7, 1024)
    return model

# === 2. Preprocess Image to 224×224 ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

# === 3. Extract and Save Features ===
def extract_features(image_dir, image_ids, weights_path, output_file):
    model = build_chexnet_feature_extractor(weights_path)
    features = {}

    for img_id in tqdm(image_ids):
        filename = img_id if img_id.endswith(".png") else img_id + ".png"
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue
        try:
            img_tensor = preprocess_image(img_path)
            fmap = model.predict(img_tensor)  # (1, 7, 7, 1024)
            fmap = np.reshape(fmap, (49, 1024))  # Flatten spatial dimensions
            features[filename] = fmap
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open(output_file, "wb") as f:
        pickle.dump(features, f)
    print(f"\n Saved feature maps for {len(features)} images → {output_file}")

# === 4. Run Extraction ===
if __name__ == "__main__":
    image_dir = "C:/Users/saich/Downloads/Research Papers/images"
    csv_path = "C:/Users/saich/Downloads/Research Papers/report_training_data.csv"
    weights_path = "C:/Users/saich/Downloads/Research Papers/CheXNet_Keras_weights.h5"
    output_file = "feature_maps.pkl"

    df = pd.read_csv(csv_path)
    df['image_id'] = df['image_id'].apply(lambda x: x if x.endswith(".png") else x + ".png")
    extract_features(image_dir, df['image_id'].tolist(), weights_path, output_file)
