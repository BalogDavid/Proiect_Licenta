package com.example.tagger;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class ScanActivity extends AppCompatActivity {

    private ImageView imageView;
    private Button captureButton;
    private Uri imageUri;
    private String brandName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scan);

        imageView = findViewById(R.id.imageView);
        captureButton = findViewById(R.id.captureButton);

        brandName = getIntent().getStringExtra("BRAND_NAME");

        captureButton.setOnClickListener(view -> openCamera());
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            try {
                File photoFile = createImageFile();
                imageUri = FileProvider.getUriForFile(this, getPackageName() + ".provider", photoFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                cameraLauncher.launch(intent);
            } catch (IOException e) {
                Toast.makeText(this, "Eroare la crearea fișierului!", Toast.LENGTH_SHORT).show();
                Log.e("ScanActivity", "Eroare: " + e.getMessage());
            }
        }
    }

    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        return File.createTempFile(imageFileName, ".jpg", storageDir);
    }

    private final ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK) {
                    imageView.setImageURI(imageUri);
                    processImageAndNavigate();
                } else {
                    Toast.makeText(this, "Capturare anulată", Toast.LENGTH_SHORT).show();
                }
            });

    private void processImageAndNavigate() {
        try {
            LabelClassifier classifier = new LabelClassifier(this);
            String result = classifier.classifyImage(imageUri.getPath());
            classifier.close();

            Intent intent = new Intent(ScanActivity.this, ResultActivity.class);
            intent.putExtra("BRAND_NAME", brandName);
            intent.putExtra("RESULT", result);
            startActivity(intent);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Eroare la clasificare!", Toast.LENGTH_SHORT).show();
        }
    }
}
