import os
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# Configuration
MODEL_FILE = 'nike_label_cnn_model.h5'
TFLITE_MODEL_FILE = 'nike_label_model.tflite'
QUANTIZED_MODEL_FILE = 'nike_label_model_quantized.tflite'
SAMPLE_IMAGE_DIR = 'Poze Reale'  # Directory with sample images for representative dataset
IMG_SIZE = (150, 150)  # Must match the size used in the original model

def load_sample_images(directory, num_samples=100):
    """Load sample images for quantization."""
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if os.path.isfile(file_path) and file_ext in valid_extensions:
            image_paths.append(file_path)
    
    # Select a subset of images if we have more than we need
    if len(image_paths) > num_samples:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:num_samples]
    
    return image_paths

def representative_dataset_gen():
    """Generate a representative dataset for quantization."""
    image_paths = load_sample_images(SAMPLE_IMAGE_DIR)
    
    for image_path in image_paths:
        try:
            # Load and preprocess the image
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            # Convert to RGB (from BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, IMG_SIZE)
            
            # Normalize
            img = img / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0).astype(np.float32)
            
            yield [img_batch]
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

def convert_to_tflite():
    """Convert a Keras model to TensorFlow Lite format."""
    print("Loading model...")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file {MODEL_FILE} not found.")
        return False
    
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Standard TFLite conversion
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(TFLITE_MODEL_FILE, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Standard TFLite model saved to {TFLITE_MODEL_FILE}")
    
    # Quantized model for better performance on mobile
    print("Converting to quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set the optimization strategy
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use representative dataset for full integer quantization
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    try:
        tflite_quantized_model = converter.convert()
        
        # Save the quantized model
        with open(QUANTIZED_MODEL_FILE, 'wb') as f:
            f.write(tflite_quantized_model)
            
        print(f"Quantized TFLite model saved to {QUANTIZED_MODEL_FILE}")
        return True
    except Exception as e:
        print(f"Error during quantization: {e}")
        print("Continuing with only the standard TFLite model")
        return True

def compare_model_sizes():
    """Compare the sizes of the original and converted models."""
    original_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_MODEL_FILE) / (1024 * 1024)
    
    print("\nModel Size Comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"TFLite model: {tflite_size:.2f} MB")
    
    if os.path.exists(QUANTIZED_MODEL_FILE):
        quantized_size = os.path.getsize(QUANTIZED_MODEL_FILE) / (1024 * 1024)
        print(f"Quantized TFLite model: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")

def verify_tflite_model():
    """Verify that the TFLite model works as expected."""
    # Load a test image
    sample_images = load_sample_images(SAMPLE_IMAGE_DIR, num_samples=1)
    if not sample_images:
        print("No sample images found for verification.")
        return
    
    test_image_path = sample_images[0]
    
    # Prepare the image
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FILE)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run the model
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Interpret the output
    probability = output[0][0]
    result = "Real" if probability > 0.5 else "Fake"
    confidence = probability if result == "Real" else 1 - probability
    
    print("\nTFLite Model Verification:")
    print(f"Test image: {test_image_path}")
    print(f"Prediction: {result} (Confidence: {confidence:.2f})")
    
    # Also verify the quantized model if it exists
    if os.path.exists(QUANTIZED_MODEL_FILE):
        try:
            # Load the quantized TFLite model
            q_interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_FILE)
            q_interpreter.allocate_tensors()
            
            # Get input and output tensors
            q_input_details = q_interpreter.get_input_details()
            q_output_details = q_interpreter.get_output_details()
            
            # For quantized model, we need to adjust the input
            input_scale, input_zero_point = q_input_details[0]["quantization"]
            if input_scale != 0:  # If quantized
                img_quantized = img / input_scale + input_zero_point
                img_quantized = img_quantized.astype(np.uint8)
            else:
                img_quantized = img
            
            # Run the model
            q_interpreter.set_tensor(q_input_details[0]['index'], img_quantized)
            q_interpreter.invoke()
            q_output = q_interpreter.get_tensor(q_output_details[0]['index'])
            
            # For quantized model, we need to dequantize the output
            output_scale, output_zero_point = q_output_details[0]["quantization"]
            if output_scale != 0:  # If quantized
                q_probability = (q_output.astype(np.float32) - output_zero_point) * output_scale
            else:
                q_probability = q_output
            
            q_result = "Real" if q_probability[0][0] > 0.5 else "Fake"
            q_confidence = q_probability[0][0] if q_result == "Real" else 1 - q_probability[0][0]
            
            print("\nQuantized TFLite Model Verification:")
            print(f"Prediction: {q_result} (Confidence: {q_confidence:.2f})")
        except Exception as e:
            print(f"Error verifying quantized model: {e}")

def main():
    """Main function."""
    print("TensorFlow version:", tf.__version__)
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file {MODEL_FILE} not found. Please train the model first.")
        return
    
    # Convert to TFLite format
    if convert_to_tflite():
        # Compare model sizes
        compare_model_sizes()
        
        # Verify models
        verify_tflite_model()
        
        print("\nConversion completed successfully.")
        print(f"Use '{TFLITE_MODEL_FILE}' or '{QUANTIZED_MODEL_FILE}' in your Android application.")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main() 