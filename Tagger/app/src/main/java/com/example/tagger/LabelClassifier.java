package com.example.tagger;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class LabelClassifier {
    private static final String TAG = "LabelClassifier";

    // Model parameters
    private static final int IMAGE_SIZE = 224;
    private GpuDelegate gpuDelegate = null;
    private Interpreter interpreter;
    private List<String> labels;
    private final Context context;
    private int numClasses = 0;
    private String modelPath;
    private String labelPath;

    // Constructor for Nike labels
    public LabelClassifier(Context context) {
        this.context = context;
        this.modelPath = "nike/nike_model.tflite";
        this.labelPath = "nike/nike_labels.txt";
        try {
            setupInterpreter();
            loadLabels();
        } catch (IOException e) {
            Log.e(TAG, "Error initializing classifier: " + e.getMessage());
        }
    }

    // Constructor for custom labels
    public LabelClassifier(Context context, String modelPath, String labelPath) {
        this.context = context;
        this.modelPath = modelPath;
        this.labelPath = labelPath;
        try {
            setupInterpreter();
            loadLabels();
        } catch (IOException e) {
            Log.e(TAG, "Error initializing classifier: " + e.getMessage());
        }
    }
    
    // Constructor that takes brand name and builds model and label paths
    public LabelClassifier(Context context, String brandName) throws IOException {
        this.context = context;
        this.modelPath = brandName.toLowerCase() + "/" + brandName.toLowerCase() + "_model.tflite";
        this.labelPath = brandName.toLowerCase() + "/" + brandName.toLowerCase() + "_labels.txt";
        Log.d(TAG, "Using model path: " + modelPath + " and labels path: " + labelPath);
        setupInterpreter();
        loadLabels();
    }

    private void setupInterpreter() throws IOException {
        Log.d(TAG, "Setting up interpreter for model: " + modelPath);
        
        // Display asset directory contents for debugging
        String[] assets = context.getAssets().list("");
        if (assets != null) {
            Log.d(TAG, "Asset directory contents:");
            for (String asset : assets) {
                Log.d(TAG, "- " + asset);
            }
        }
        
        String[] nikeAssets = context.getAssets().list("nike");
        if (nikeAssets != null) {
            Log.d(TAG, "Nike asset directory contents:");
            for (String asset : nikeAssets) {
                Log.d(TAG, "- " + asset);
            }
        }

        // Setup interpreter options
        Interpreter.Options options = new Interpreter.Options();
        
        // Skip GPU for compatibility and use CPU with XNNPACK
        Log.d(TAG, "Using CPU with XNNPACK acceleration");
        options.setUseXNNPACK(true);
        options.setNumThreads(4);

        try {
            MappedByteBuffer modelData = loadModelFile(context.getAssets(), modelPath);
            Log.d(TAG, "Model data loaded, size: " + modelData.capacity() + " bytes");
            interpreter = new Interpreter(modelData, options);
            
            // Log tensor info for debugging
            int[] inputShape = interpreter.getInputTensor(0).shape();
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            Log.d(TAG, "Input tensor shape: [" + inputShape[0] + ", " + inputShape[1] + ", " 
                  + inputShape[2] + ", " + inputShape[3] + "]");
            Log.d(TAG, "Output tensor shape: [" + outputShape[0] + ", " + outputShape[1] + "]");
            
            numClasses = outputShape[1]; // Get number of classes from output tensor
            Log.d(TAG, "Number of classes detected: " + numClasses);
            
        } catch (IOException e) {
            Log.e(TAG, "Error loading model: " + e.getMessage(), e);
            throw e;
        }
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            Log.d(TAG, "Loading model from " + modelPath + ", offset: " + startOffset + ", length: " + declaredLength);
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            Log.e(TAG, "Failed to load model file: " + modelPath, e);
            throw e;
        }
    }

    private void loadLabels() throws IOException {
        Log.d(TAG, "Loading labels from: " + labelPath);
        labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.length() > 0) {
                labels.add(line);
                Log.d(TAG, "Loaded label: " + line);
            }
        }
        reader.close();
        
        // Ensure labels count matches model output
        if (numClasses > 0 && labels.size() != numClasses) {
            Log.w(TAG, "Warning: Number of labels (" + labels.size() + 
                 ") doesn't match model output classes (" + numClasses + ")");
        }
    }
    
    // Method to classify an image from file path
    public String classifyImage(String imagePath) throws IOException {
        Log.d(TAG, "Loading image from: " + imagePath);
        File imgFile = new File(imagePath);
        if (!imgFile.exists()) {
            throw new IOException("Image file does not exist: " + imagePath);
        }
        
        // Load the image file as bitmap
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
        if (bitmap == null) {
            throw new IOException("Failed to decode image: " + imagePath);
        }
        
        // Call the existing classify method with the bitmap
        return classify(bitmap);
    }

    public String classify(Bitmap bitmap) {
        try {
            if (bitmap == null) {
                Log.e(TAG, "Bitmap is null");
                return "Error: No image provided";
            }
            
            if (interpreter == null) {
                Log.e(TAG, "Interpreter is null");
                return "Error: Model not loaded";
            }

            // Resize bitmap to match model input
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
            
            // Get input shape from the model to ensure correct buffer size
            int[] inputShape = interpreter.getInputTensor(0).shape();
            int batchSize = inputShape[0];
            int inputHeight = inputShape[1];
            int inputWidth = inputShape[2];
            int channels = inputShape[3];
            
            Log.d(TAG, "Model expects input shape: [" + batchSize + ", " + inputHeight + ", " + 
                  inputWidth + ", " + channels + "]");
            
            // Calculate buffer size based on model input shape (4 bytes per float)
            int bufferSize = batchSize * inputHeight * inputWidth * channels * 4;
            Log.d(TAG, "Creating buffer with size: " + bufferSize + " bytes");
            
            // Prepare input data with correct size
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(bufferSize);
            inputBuffer.order(ByteOrder.nativeOrder());
            
            int[] pixels = new int[inputHeight * inputWidth];
            resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);
            
            // Normalize pixel values to [0, 1]
            for (int pixel : pixels) {
                // Extract RGB values
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;
                
                // TensorFlow model expects RGB inputs
                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }
            
            // Reset position to start
            inputBuffer.rewind();
            
            // Prepare output array
            float[][] outputBuffer = new float[1][numClasses > 0 ? numClasses : labels.size()];
            
            // Run inference
            interpreter.run(inputBuffer, outputBuffer);
            
            // Get prediction result
            int maxIndex = 0;
            float maxConfidence = 0;
            
            Log.d(TAG, "Classification results:");
            for (int i = 0; i < outputBuffer[0].length; i++) {
                float confidence = outputBuffer[0][i];
                if (i < labels.size()) {
                    Log.d(TAG, labels.get(i) + ": " + confidence);
                }
                
                if (confidence > maxConfidence) {
                    maxConfidence = confidence;
                    maxIndex = i;
                }
            }
            
            if (maxIndex >= labels.size()) {
                Log.e(TAG, "Max index out of bounds: " + maxIndex + ", labels size: " + labels.size());
                return "Error: Invalid classification result";
            }
            
            // Formatează rezultatul pentru a include eticheta și scorul în formatul așteptat de ResultActivity
            String label = labels.get(maxIndex);
            String result = label + " (" + maxConfidence + ")";
            Log.d(TAG, "Best match: " + result);
            return result;
            
        } catch (Exception e) {
            Log.e(TAG, "Error classifying image: " + e.getMessage(), e);
            return "Error: " + e.getMessage();
        }
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }
    }
}
