import os
import subprocess
import sys
import time

def check_requirements():
    """Check if required packages are installed."""
    try:
        import numpy
        import cv2
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        print("All required packages are installed.")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        return False

def train_model():
    """Run the CNN model training script."""
    print("\n" + "="*50)
    print("STEP 1: Training CNN Model")
    print("="*50)
    
    cnn_script = "nike_label_cnn_authenticator.py"
    if not os.path.exists(cnn_script):
        print(f"Error: {cnn_script} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, cnn_script], 
                               check=True, 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check if model was created
        if os.path.exists("nike_label_cnn_model.h5"):
            print("CNN model trained successfully!")
            return True
        else:
            print("Model training failed: output file not found.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running training script: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def convert_to_tflite():
    """Run the TFLite conversion script."""
    print("\n" + "="*50)
    print("STEP 2: Converting to TensorFlow Lite")
    print("="*50)
    
    tflite_script = "nike_label_tflite_converter.py"
    if not os.path.exists(tflite_script):
        print(f"Error: {tflite_script} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, tflite_script], 
                               check=True, 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check if tflite model was created
        if os.path.exists("nike_label_model.tflite"):
            print("TensorFlow Lite conversion completed successfully!")
            return True
        else:
            print("TFLite conversion failed: output file not found.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running conversion script: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    """Main function to run the entire pipeline."""
    start_time = time.time()
    
    print("Nike Label Authenticator - Training and Conversion Pipeline")
    print("="*70 + "\n")
    
    # Check for required packages
    if not check_requirements():
        print("Please install the required packages:")
        print("pip install numpy opencv-python scikit-learn tensorflow matplotlib")
        return
    
    # Check for image directories
    if not os.path.exists("Poze Fake") or not os.path.exists("Poze Reale"):
        print("Error: Image directories 'Poze Fake' and 'Poze Reale' must exist.")
        return
    
    # Run the training step
    if not train_model():
        print("Training step failed. Pipeline stopped.")
        return
    
    # Run the conversion step
    if not convert_to_tflite():
        print("Conversion step failed.")
    
    # Done
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Pipeline completed in {elapsed_time/60:.2f} minutes")
    print("="*50)
    
    print("\nOutput files:")
    for output_file in ["nike_label_cnn_model.h5", "nike_label_model.tflite", "nike_label_model_quantized.tflite"]:
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"- {output_file} ({size_mb:.2f} MB)")
    
    print("\nVisualization files:")
    for viz_file in ["training_history.png", "prediction_samples.png"]:
        if os.path.exists(viz_file):
            print(f"- {viz_file}")
    
    print("\nFor Android integration, use 'nike_label_model.tflite' or 'nike_label_model_quantized.tflite'")
    print("Refer to README.md for more details.")

if __name__ == "__main__":
    main() 