import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
FAKE_DIR = 'Poze Fake'
REAL_DIR = 'Poze Reale'
IMG_SIZE = (150, 150)  # Size for CNN input
MODEL_FILE = 'nike_label_cnn_model.h5'
BATCH_SIZE = 32
EPOCHS = 20

def load_and_preprocess_data():
    """Load and preprocess all images from both directories."""
    # Lists to store data
    images = []
    labels = []
    filenames = []
    
    # Valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # Load fake images
    print("Loading fake labels...")
    for filename in os.listdir(FAKE_DIR):
        file_path = os.path.join(FAKE_DIR, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: Could not load {file_path}")
                    continue
                
                # Convert to RGB (from BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, IMG_SIZE)
                
                # Normalize
                img = img / 255.0
                
                # Add to dataset
                images.append(img)
                labels.append(0)  # 0 = fake
                filenames.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Load real images
    print("Loading real labels...")
    for filename in os.listdir(REAL_DIR):
        file_path = os.path.join(REAL_DIR, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: Could not load {file_path}")
                    continue
                
                # Convert to RGB (from BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, IMG_SIZE)
                
                # Normalize
                img = img / 255.0
                
                # Add to dataset
                images.append(img)
                labels.append(1)  # 1 = real
                filenames.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return np.array(images), np.array(labels), filenames

def create_model():
    """Create a CNN model for label classification."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (fake/real)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the CNN model and save it."""
    # Load and preprocess data
    X, y, _ = load_and_preprocess_data()
    
    print(f"Dataset: {len(X)} images ({len(y) - sum(y)} fake, {sum(y)} real)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create an image data generator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create the model
    model = create_model()
    
    # Display model summary
    model.summary()
    
    # Early stopping and model checkpoint callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training history saved as 'training_history.png'")
    
    # Save the model
    model.save(MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}")
    
    return model, history

def predict_image(image_path, model):
    """Predict if a label image is real or fake using the CNN model."""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not load image", 0.0
        
        # Convert to RGB (from BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        
        # Normalize
        img = img / 255.0
        
        # Expand dimensions to create batch of size 1
        img = np.expand_dims(img, axis=0)
        
        # Predict
        probability = model.predict(img)[0][0]
        
        # The model outputs probability of being real
        # >0.5 means real, <0.5 means fake
        if probability > 0.5:
            result = "Real"
            confidence = probability
        else:
            result = "Fake"
            confidence = 1 - probability
        
        return result, float(confidence)
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def visualize_layers_activations(model, image_path):
    """Visualize activations of the convolutional layers."""
    # Load and preprocess image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize
    img = img / 255.0
    
    # Expand dimensions to create batch of size 1
    img = np.expand_dims(img, axis=0)
    
    # Get outputs of all conv layers
    layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)
    
    # Plot the activations
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(len(activations) + 1, 1, 1)
    plt.imshow(img[0])
    plt.title('Original Image')
    plt.axis('off')
    
    # For each conv layer, plot a subset of activation channels
    for i, layer_activations in enumerate(activations):
        num_channels = min(8, layer_activations.shape[-1])
        for j in range(num_channels):
            plt.subplot(len(activations) + 1, num_channels, (i+1)*num_channels + j + 1)
            plt.imshow(layer_activations[0, :, :, j], cmap='viridis')
            plt.title(f'Layer {i+1}, Channel {j+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('layer_activations.png')
    plt.close()
    
    print("Layer activations saved as 'layer_activations.png'")

def visualize_predictions(model):
    """Visualize predictions on a sample of test images."""
    # Load all images
    X, y, filenames = load_and_preprocess_data()
    
    # Split into training and test sets
    _, X_test, _, y_test, _, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select a random subset of test images
    num_samples = min(10, len(X_test))
    indices = random.sample(range(len(X_test)), num_samples)
    
    # Create figure for plotting
    plt.figure(figsize=(15, 10))
    
    # For each sample
    for i, idx in enumerate(indices):
        # Get the image and true label
        img = X_test[idx]
        true_label = "Real" if y_test[idx] == 1 else "Fake"
        
        # Make prediction
        img_batch = np.expand_dims(img, axis=0)
        probability = model.predict(img_batch)[0][0]
        
        # Interpret prediction
        if probability > 0.5:
            predicted_label = "Real"
            confidence = probability
        else:
            predicted_label = "Fake"
            confidence = 1 - probability
        
        # Plot the image
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        
        # Set title color based on correctness
        title_color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {predicted_label} ({confidence:.2f})", 
                  color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.close()
    
    print("Prediction samples saved as 'prediction_samples.png'")

def main():
    """Main function."""
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print("Training new model...")
        model, _ = train_model()
    else:
        # Ask if user wants to retrain
        retrain = input("Model already exists. Do you want to retrain? (y/n): ")
        if retrain.lower() == 'y':
            model, _ = train_model()
        else:
            # Load existing model
            print("Loading existing model...")
            model = load_model(MODEL_FILE)
    
    # Visualize predictions
    print("Visualizing predictions on test samples...")
    visualize_predictions(model)
    
    # Interactive prediction
    while True:
        file_path = input("\nEnter path to image to test (or 'q' to quit, 'v' to visualize layers): ")
        
        if file_path.lower() == 'q':
            break
        
        elif file_path.lower() == 'v':
            vis_path = input("Enter path to image for layer visualization: ")
            if os.path.exists(vis_path):
                visualize_layers_activations(model, vis_path)
            else:
                print(f"File not found: {vis_path}")
        
        elif os.path.exists(file_path):
            result, confidence = predict_image(file_path, model)
            print(f"Prediction: {result} (Confidence: {confidence:.2f})")
        
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Run main function
    main() 