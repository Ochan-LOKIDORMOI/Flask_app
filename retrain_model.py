import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import save_model # type: ignore # for saving the model in h5 format

IMG_SIZE = 150

def preprocess_user_data(directory, categories):
    data = []

    for category in categories:
        path = os.path.join(directory, category)
        label = categories.index(category)

        if not os.path.isdir(path):
            print(f"Skipping invalid category path: {path}")
            continue

        has_images = False  # Track if valid images are found
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    print(f"Skipping non-image file: {img_path}")
                    continue

                img_arr = cv2.imread(img_path)
                if img_arr is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([img_arr, label])
                has_images = True
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

        if not has_images:
            print(f"No valid images found in category: {category}")

    if not data:
        raise ValueError("No valid image data found in the provided dataset.")

    random.shuffle(data)
    X, y = [], []
    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X) / 255.0
    y = np.array(y)

    return X, y

def retrain_model(user_data_dir, categories):
    print("Preprocessing user data...")
    X, y = preprocess_user_data(user_data_dir, categories)
    print(f"Preprocessed data shapes: X={X.shape}, y={y.shape}")

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # 10% for test
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 22.5% for val

    # Build the model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), verbose=1)

    print("Retrained model successfully!")

    # Save the model
    model_save_path = r"C:\Users\HP\Desktop\Summative_assignment-MLOP\models\retrain.h5"
    save_model(model, model_save_path)
 # Save the model as h5 file

    print(f"Model saved as {model_save_path}")
    return model
