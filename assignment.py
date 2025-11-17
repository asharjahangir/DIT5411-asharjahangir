import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

# Set UTF-8 encoding for console output
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configuration - CHANGE THESE VALUES AS NEEDED
MAX_CHARACTERS = 40  # Limit number of characters to process
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 3  # Reduced for testing
AUGMENTED_SAMPLES_PER_CLASS = 50  # Reduced for testing

def safe_imread(file_path):
    """Safely read image files with Chinese characters in path using np.fromfile"""
    try:
        # Read file as binary and decode with OpenCV
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return image
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_and_preprocess_data(data_path, max_characters=MAX_CHARACTERS):
    print("Loading and preprocessing data...")
    
    # Collect all image files and group by character
    character_images = {}
    
    for filename in os.listdir(data_path):
        if filename.endswith('.png'):
            # Extract character name from filename (remove _number.png)
            character_name = filename.split('_')[0]
            
            if character_name not in character_images:
                character_images[character_name] = []
            
            character_images[character_name].append(filename)
    
    print(f"Found {len(character_images)} unique characters")
    
    # Filter characters that have at least 50 samples (40 for training + 10 for testing)
    valid_characters = {char: files for char, files in character_images.items() 
                       if len(files) >= 50}
    
    print(f"Using {len(valid_characters)} characters with sufficient samples")
    
    # Limit number of characters to process
    limited_characters = dict(list(valid_characters.items())[:max_characters])
    print(f"Limited to {len(limited_characters)} characters for processing")
    
    return limited_characters

def split_train_test(character_images):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    for character, files in character_images.items():
        # Sort files to maintain consistency
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # First 40 for training, rest for testing
        train_files = files[:40]
        test_files = files[40:50]  # Use next 10 for testing
        
        # Add to training set
        for file in train_files:
            train_images.append(file)
            train_labels.append(character)
        
        # Add to testing set
        for file in test_files:
            test_images.append(file)
            test_labels.append(character)
    
    return train_images, train_labels, test_images, test_labels

def augment_image(image):
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Rotation
    for angle in [-5, 5]:
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
    
    # Scaling
    for scale in [0.95, 1.05]:
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
        scaled = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(scaled)
    
    # Shearing
    for shear in [-0.05, 0.05]:
        rows, cols = image.shape[:2]
        M = np.float32([[1, shear, 0], [shear, 1, 0]])
        sheared = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(sheared)
    
    # Brightness adjustment
    for beta in [-20, 20]:
        bright = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        augmented_images.append(bright)
    
    return augmented_images

def create_augmented_dataset(train_images, train_labels, data_path):
    print("Creating augmented dataset...")
    
    augmented_images = []
    augmented_labels = []
    
    # Group by character
    character_groups = {}
    for img, label in zip(train_images, train_labels):
        if label not in character_groups:
            character_groups[label] = []
        character_groups[label].append(img)
    
    # Augment each character's images
    total_characters = len(character_groups)
    current_character = 0
    
    for character, images in character_groups.items():
        current_character += 1
        character_augmented = []
        
        # Load and augment each original image
        successful_loads = 0
        for img_file in images:
            img_path = os.path.join(data_path, img_file)
            img = safe_imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                augmented = augment_image(img)
                character_augmented.extend(augmented)
                successful_loads += 1
        
        print(f"Character {current_character}/{total_characters}: {successful_loads}/{len(images)} images loaded successfully")
        
        # If we need more samples, randomly select from augmented ones
        while len(character_augmented) < AUGMENTED_SAMPLES_PER_CLASS:
            if character_augmented:
                character_augmented.extend(random.sample(character_augmented, 
                                                       min(len(character_augmented), 
                                                           AUGMENTED_SAMPLES_PER_CLASS - len(character_augmented))))
            else:
                # If no images were loaded, create a blank image
                blank_img = np.ones(IMAGE_SIZE, dtype=np.uint8) * 255
                character_augmented.append(blank_img)
        
        # Take exactly the required number of samples
        character_augmented = character_augmented[:AUGMENTED_SAMPLES_PER_CLASS]
        
        augmented_images.extend(character_augmented)
        augmented_labels.extend([character] * len(character_augmented))
        
        # Clear memory periodically
        if current_character % 100 == 0:
            print(f"Processed {current_character} characters...")
    
    return augmented_images, augmented_labels

def prepare_final_data(augmented_images, augmented_labels, test_images, test_labels, data_path):
    print("Preparing final datasets...")
    
    # Load and preprocess test images
    X_test = []
    y_test = []
    
    successful_test_loads = 0
    total_test_files = len(test_images)
    
    for i, (img_file, label) in enumerate(zip(test_images, test_labels)):
        img_path = os.path.join(data_path, img_file)
        img = safe_imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            X_test.append(img)
            y_test.append(label)
            successful_test_loads += 1
        
        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{total_test_files} test images...")
    
    print(f"Successfully loaded {successful_test_loads}/{total_test_files} test images")
    
    # Convert to numpy arrays in batches to save memory
    print("Converting training data to numpy arrays...")
    
    # Process training data in smaller chunks
    chunk_size = 10000
    X_train_chunks = []
    
    for i in range(0, len(augmented_images), chunk_size):
        chunk = augmented_images[i:i + chunk_size]
        chunk_array = np.array(chunk, dtype='float32') / 255.0
        X_train_chunks.append(chunk_array)
        print(f"Processed training chunk {i//chunk_size + 1}/{(len(augmented_images)-1)//chunk_size + 1}")
    
    X_train = np.concatenate(X_train_chunks, axis=0)
    y_train = np.array(augmented_labels)
    
    # Process test data
    print("Converting test data to numpy arrays...")
    X_test = np.array(X_test, dtype='float32') / 255.0
    y_test = np.array(y_test)
    
    # Clear memory
    del augmented_images
    del X_train_chunks
    
    # Add channel dimension
    X_train = X_train.reshape(X_train.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    X_test = X_test.reshape(X_test.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    
    # Encode labels - this converts Chinese characters to numeric labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to categorical
    num_classes = len(label_encoder.classes_)
    y_train_categorical = keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_categorical = keras.utils.to_categorical(y_test_encoded, num_classes)
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Number of classes: {num_classes}")
    
    return X_train, X_test, y_train_categorical, y_test_categorical, num_classes, label_encoder

def create_model_1(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_model_2(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_model_3(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Add callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        verbose=1)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    
    return test_accuracy, history

def main():
    data_path = "sampleimages"
    
    # Verify the data path exists
    if not os.path.exists(data_path):
        print(f"Error: Directory '{data_path}' does not exist!")
        return
    
    # Load and preprocess data
    character_images = load_and_preprocess_data(data_path)
    
    if not character_images:
        print("No valid character data found!")
        return
    
    # Split into train and test
    train_images, train_labels, test_images, test_labels = split_train_test(character_images)
    
    print(f"Training images: {len(train_images)}")
    print(f"Testing images: {len(test_images)}")
    
    # Create augmented dataset
    augmented_images, augmented_labels = create_augmented_dataset(train_images, train_labels, data_path)
    
    # Prepare final datasets
    X_train, X_test, y_train, y_test, num_classes, label_encoder = prepare_final_data(
        augmented_images, augmented_labels, test_images, test_labels, data_path)
    
    print(f"\nFinal dataset prepared:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of classes: {num_classes}")
    
    # Define input shape
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    
    # Create and train multiple models
    models = {
        "Model 1 (Simple CNN)": create_model_1(input_shape, num_classes),
        "Model 2 (Deep CNN)": create_model_2(input_shape, num_classes),
        "Model 3 (Wider CNN)": create_model_3(input_shape, num_classes)
    }
    
    best_accuracy = 0
    best_model_name = ""
    best_model = None
    results = {}
    
    for name, model in models.items():
        accuracy, history = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = {
            'accuracy': accuracy,
            'history': history,
            'model': model
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
    
    print(f"\n=== RESULTS ===")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save the best model
    best_model.save("best_chinese_character_model.h5")
    print("Best model saved as 'best_chinese_character_model.h5'")

if __name__ == "__main__":
    main()