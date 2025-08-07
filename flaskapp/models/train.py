import os
import zipfile
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Function to download and extract dataset
def download_and_extract(url, extract_to='datasets'):
    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(extract_to, local_filename)
    
    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Extract the file
    with zipfile.ZipFile(local_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Clean up the zip file
    os.remove(local_filepath)

# Download and extract the DIV2K dataset
div2k_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
download_and_extract(div2k_url)

# Load high-resolution images
def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

# Load high-resolution images
high_res_folder = 'datasets/DIV2K_train_HR/'
X_train = load_images_from_folder(high_res_folder)

# Normalize images to the range [0, 1]
X_train = X_train / 255.0

# Define the model architecture
def build_model(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
    return model

# Build and train the model
model = build_model()
checkpoint = ModelCheckpoint('enhancer_model.h5', save_best_only=True, monitor='loss', mode='min')
history = model.fit(X_train, X_train, epochs=20, batch_size=16, validation_split=0.2, callbacks=[checkpoint])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Save the final model
model.save('enhancer_model_final.h5')
