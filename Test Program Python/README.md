# Nike Label Authenticator

This project provides machine learning models to authenticate Nike product labels by distinguishing between real and fake labels. The system was developed as a proof of concept for an Android application.

## Project Structure

- `nike_label_authenticator.py`: A simple ML model using Random Forest for label classification
- `nike_label_cnn_authenticator.py`: An advanced model using Convolutional Neural Networks (CNN) for better accuracy
- `nike_label_tflite_converter.py`: A tool to convert the CNN model to TensorFlow Lite format for Android compatibility
- `Poze Fake/`: Directory containing images of fake Nike labels
- `Poze Reale/`: Directory containing images of real Nike labels

## Requirements

Install the required packages:

```bash
pip install numpy opencv-python scikit-learn tensorflow matplotlib
```

## Usage Instructions

### Basic Random Forest Model

Run the basic authenticator:

```bash
python nike_label_authenticator.py
```

This will:
1. Train a Random Forest model on the label images
2. Evaluate the model's performance
3. Save the model as `nike_label_model.pkl`
4. Allow you to test new images interactively

### Advanced CNN Model (Recommended)

Run the CNN-based authenticator:

```bash
python nike_label_cnn_authenticator.py
```

This will:
1. Train a Convolutional Neural Network on the label images
2. Apply data augmentation to improve generalization
3. Evaluate the model's performance with visual reports
4. Save the model as `nike_label_cnn_model.h5`
5. Visualize model performance on test samples
6. Allow you to test new images interactively

Additional options in the CNN model:
- Enter `v` when prompted to visualize layer activations on a specific image
- The program generates visualization files:
  - `training_history.png`: Training/validation accuracy and loss curves
  - `prediction_samples.png`: Model predictions on test samples
  - `layer_activations.png`: Visualization of CNN layer activations (when requested)

### Preparing for Android Integration

Convert the CNN model to TensorFlow Lite format:

```bash
python nike_label_tflite_converter.py
```

This will:
1. Convert the CNN model to TensorFlow Lite format
2. Create a quantized version for better mobile performance
3. Verify the model's functionality
4. Output models:
   - `nike_label_model.tflite`: Standard TFLite model
   - `nike_label_model_quantized.tflite`: Optimized model for mobile devices

## Model Performance

The CNN model typically achieves 90%+ accuracy on the test set. Performance may vary based on:
- Quality and diversity of images in the training set
- Similarity between fake and real labels
- Lighting conditions and image quality

## Android Integration

To integrate the model into an Android application:
1. Use the generated `.tflite` file in your Android project
2. Follow TensorFlow Lite Android integration guidelines
3. Process images using the same preprocessing steps:
   - Resize to 150x150 pixels
   - Convert to RGB format
   - Normalize pixel values to [0,1]

## Adding More Training Data

To improve the model:
1. Add more images to the respective folders (`Poze Fake/` and `Poze Reale/`)
2. Retrain the model by running the script again
3. When prompted "Model already exists. Do you want to retrain?", answer 'y'

## License

This project is for educational purposes only. 