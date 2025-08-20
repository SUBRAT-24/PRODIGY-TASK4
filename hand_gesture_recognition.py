import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10  # Reduced for faster training

def load_data_from_directory(data_dir):
    """Load images and labels from the dataset directory"""
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        print(f"Loading images from {class_name}...")
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Read and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0  # Normalize to [0,1]
                    
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    if not images:
        raise ValueError("No images found in the dataset directory")
    
    return np.array(images), np.array(labels), class_names

def create_model(input_shape, num_classes):
    """Create a CNN model for hand gesture recognition"""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Set path to your dataset directory
    # The directory should contain subdirectories for each class
    # Example structure:
    # dataset/
    #   ├── class1/
    #   ├── class2/
    #   └── ...
    data_dir = "dataset"  # Change this to your dataset directory
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        print("Please create a 'dataset' directory with subdirectories for each class.")
        return
    
    print("Loading dataset...")
    X, y, class_names = load_data_from_directory(data_dir)
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=len(class_names))
    
    # Reshape X to include channel dimension
    X = np.expand_dims(X, axis=-1)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {', '.join(class_names)}")
    
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create and compile model
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = create_model(input_shape, len(class_names))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Generate predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Plot training history
    plot_training_history(history)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save the model
    model.save('hand_gesture_model.h5')
    print("\nModel saved as 'hand_gesture_model.h5'")

if __name__ == "__main__":
    main()
