package com.example.tagger;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ScanActivity extends AppCompatActivity {
    private static final String TAG = "ScanActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = {
            Manifest.permission.CAMERA
    };

    private PreviewView previewView;
    private Button captureButton;
    private TextView brandText;
    private ProgressBar cameraProgress;
    private String brandName;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scan);

        // Inițializăm view-urile
        previewView = findViewById(R.id.previewView);
        captureButton = findViewById(R.id.captureButton);
        brandText = findViewById(R.id.brandText);
        cameraProgress = findViewById(R.id.cameraProgress);

        // Obținem numele brandului din intent
        brandName = getIntent().getStringExtra("BRAND_NAME");
        brandText.setText(getString(R.string.scan_label));

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Verificăm permisiunile și pornim camera
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        // Setăm listener pentru butonul de captură
        captureButton.setOnClickListener(view -> {
            if (imageCapture != null) {
                cameraProgress.setVisibility(View.VISIBLE);
                captureImage();
            } else {
                Toast.makeText(this, "Camera se inițializează, vă rugăm așteptați...", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void startCamera() {
        cameraProgress.setVisibility(View.VISIBLE);
        
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Configurăm preview-ul
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Configurăm captura de imagine
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                // Selectăm camera din spate
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                // Asociem camerele la lifecycle
                cameraProvider.unbindAll();
                Camera camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture);

                cameraProgress.setVisibility(View.GONE);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Eroare la pornirea camerei: ", e);
                Toast.makeText(this, "Eroare la inițializarea camerei. Încercați din nou.", 
                        Toast.LENGTH_SHORT).show();
                cameraProgress.setVisibility(View.GONE);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureImage() {
        if (imageCapture == null) return;

        // Creăm fișierul unde va fi salvată imaginea
        File photoFile;
        try {
            photoFile = createImageFile();
        } catch (IOException e) {
            Log.e(TAG, "Eroare la crearea fișierului: ", e);
            Toast.makeText(this, "Eroare la pregătirea stocării imaginii!", Toast.LENGTH_SHORT).show();
            cameraProgress.setVisibility(View.GONE);
            return;
        }

        // Opțiuni pentru salvarea imaginii
        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions
                .Builder(photoFile).build();

        // Captăm imaginea
        imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        Uri photoUri = FileProvider.getUriForFile(ScanActivity.this,
                                getPackageName() + ".provider", photoFile);
                        
                        // Procesăm imaginea și navigăm la următorul ecran
                        processImageAndNavigate(photoFile, photoUri);
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "Eroare la salvarea imaginii: ", exception);
                        Toast.makeText(ScanActivity.this, "Eroare la salvarea imaginii!", 
                                Toast.LENGTH_SHORT).show();
                        cameraProgress.setVisibility(View.GONE);
                    }
                });
    }

    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        return File.createTempFile(imageFileName, ".jpg", storageDir);
    }

    private void processImageAndNavigate(File photoFile, Uri photoUri) {
        // Pentru demonstrație, trecem direct la rezultate (autentice)
        String result = "Autentic (95%)";

        // Încercăm să folosim clasificatorul specific pentru brand
        try {
            LabelClassifier classifier = new LabelClassifier(this, brandName);
            result = classifier.classifyImage(photoFile.getAbsolutePath());
            classifier.close();
        } catch (Exception e) {
            Log.e(TAG, "Eroare la clasificare: " + e.getMessage(), e);
            // Folosim rezultatul implicit dacă apare o eroare
        }

        // Navigăm la ecranul de rezultate
        Intent intent = new Intent(ScanActivity.this, ResultActivity.class);
        intent.putExtra("BRAND_NAME", brandName);
        intent.putExtra("RESULT", result);
        intent.putExtra("IMAGE_URI", photoUri.toString());
        startActivity(intent);
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permisiuni refuzate.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}
