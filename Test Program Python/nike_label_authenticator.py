import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
FAKE_DIR = 'Poze Fake'
REAL_DIR = 'Poze Reale'
IMG_SIZE = (128, 128)  # Resize images to this size
MODEL_FILE = 'nike_label_model.pkl'

def load_images_from_dir(directory, label):
    """Load images from directory and assign label."""
    images = []
    labels = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            try:
                # Load image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: Could not load {file_path}")
                    continue
                    
                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize
                img = cv2.resize(img, IMG_SIZE)
                
                # Normalize pixel values
                img = img / 255.0
                
                # Add to dataset
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return images, labels

def extract_features(images):
    """Extract features from images."""
    # Simple feature extraction - flatten pixels
    features = [img.flatten() for img in images]
    
    # Add more sophisticated features here as needed
    # For example: HOG features, SIFT, SURF, etc.
    
    return np.array(features)

def train_model():
    """Train model on label images and save it."""
    print("Loading fake labels...")
    fake_images, fake_labels = load_images_from_dir(FAKE_DIR, 0)  # 0 = fake
    
    print("Loading real labels...")
    real_images, real_labels = load_images_from_dir(REAL_DIR, 1)  # 1 = real
    
    # Combine datasets
    all_images = fake_images + real_images
    all_labels = fake_labels + real_labels
    
    print(f"Dataset: {len(all_images)} images ({len(fake_images)} fake, {len(real_images)} real)")
    
    # Extract features
    print("Extracting features...")
    features = extract_features(all_images)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, all_labels, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model and scaler
    print("Saving model...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump((model, scaler), f)
    
    return model, scaler, accuracy

def predict_image(image_path, model, scaler):
    """Predict if a label image is real or fake."""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not load image"
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        
        # Extract features
        features = img.flatten().reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = "Real" if prediction == 1 else "Fake"
        confidence = probability[prediction]
        
        return result, confidence
    
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def evaluate_with_visualization(model, scaler):
    """Evaluate model on random test images and visualize results."""
    # Load some test images
    fake_images, _ = load_images_from_dir(FAKE_DIR, 0)
    real_images, _ = load_images_from_dir(REAL_DIR, 1)
    
    # Get file lists
    fake_files = [f for f in os.listdir(FAKE_DIR) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    real_files = [f for f in os.listdir(REAL_DIR) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    
    # Select random samples for testing
    np.random.seed(42)
    fake_samples = np.random.choice(fake_files, min(5, len(fake_files)), replace=False)
    real_samples = np.random.choice(real_files, min(5, len(real_files)), replace=False)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    for i, img_file in enumerate(list(fake_samples) + list(real_samples)):
        # Determine true class
        is_real = img_file in real_samples
        img_dir = REAL_DIR if is_real else FAKE_DIR
        true_label = "Real" if is_real else "Fake"
        
        # Make prediction
        result, confidence = predict_image(os.path.join(img_dir, img_file), model, scaler)
        
        # Load original image for display
        img = cv2.imread(os.path.join(img_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        
        # Set title color based on correctness
        title_color = 'green' if result == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {result} ({confidence:.2f})", 
                  color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()
    
    print("Results visualization saved as 'evaluation_results.png'")

def predict_from_file(file_path):
    """Load model and predict from a single file."""
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Training new model...")
        model, scaler, _ = train_model()
    else:
        # Load model
        with open(MODEL_FILE, 'rb') as f:
            model, scaler = pickle.load(f)
    
    # Make prediction
    result, confidence = predict_image(file_path, model, scaler)
    return result, confidence

def main():
    """Main function."""
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print("Training new model...")
        model, scaler, _ = train_model()
    else:
        # Ask if user wants to retrain
        retrain = input("Model already exists. Do you want to retrain? (y/n): ")
        if retrain.lower() == 'y':
            model, scaler, _ = train_model()
        else:
            # Load existing model
            with open(MODEL_FILE, 'rb') as f:
                model, scaler = pickle.load(f)
    
    # Evaluate with visualization
    evaluate_with_visualization(model, scaler)
    
    # Interactive prediction
    while True:
        file_path = input("\nEnter path to image to test (or 'q' to quit): ")
        if file_path.lower() == 'q':
            break
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        result, confidence = predict_image(file_path, model, scaler)
        print(f"Prediction: {result} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main() 