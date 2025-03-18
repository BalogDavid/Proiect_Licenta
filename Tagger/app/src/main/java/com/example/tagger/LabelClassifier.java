package com.example.tagger;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

public class LabelClassifier {
    private static final String MODEL_NAME = "model.tflite";  // Asigură-te că ai acest nume în assets
    private static final int IMAGE_SIZE = 224;  // Ajustează în funcție de model

    private Interpreter interpreter;
    private List<String> labels;

    public LabelClassifier(Context context) throws IOException {
        MappedByteBuffer tfliteModel = loadModelFile(context);
        interpreter = new Interpreter(tfliteModel);
        labels = FileUtil.loadLabels(context, "labels.txt"); // Adaugă și fișierul labels.txt în assets
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        File file = new File(context.getFilesDir(), MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(file);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length());
    }

    public String classifyImage(String imagePath) {
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);

        TensorImage tensorImage = new TensorImage();
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        tensorImage.load(resizedBitmap);
        tensorImage = imageProcessor.process(tensorImage);

        float[][] output = new float[1][labels.size()];
        interpreter.run(tensorImage.getBuffer(), output);

        int maxIndex = 0;
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > output[0][maxIndex]) {
                maxIndex = i;
            }
        }

        return labels.get(maxIndex) + " (" + (output[0][maxIndex] * 100) + "%)";
    }

    public void close() {
        interpreter.close();
    }
}
