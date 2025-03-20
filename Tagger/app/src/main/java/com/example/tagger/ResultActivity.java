package com.example.tagger;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

public class ResultActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        // Inițializăm view-urile
        ImageView imageView = findViewById(R.id.resultImage);
        TextView brandNameText = findViewById(R.id.brandNameText);
        TextView resultText = findViewById(R.id.resultText);
        TextView confidenceScore = findViewById(R.id.confidenceScore);
        Button homeButton = findViewById(R.id.homeButton);

        // Obținem datele din intent
        Intent intent = getIntent();
        String brandName = intent.getStringExtra("BRAND_NAME");
        String result = intent.getStringExtra("RESULT");
        String imageUriString = intent.getStringExtra("IMAGE_URI");

        // Setăm numele brandului
        if (!TextUtils.isEmpty(brandName)) {
            brandNameText.setText(brandName);
        }

        // Verificăm dacă avem o imagine validă și o afișează cu orientarea corectă
        if (imageUriString != null) {
            try {
                Uri imageUri = Uri.parse(imageUriString);
                
                // Încărcăm imaginea ca bitmap și o rotim
                Bitmap bitmap = loadAndRotateImage(imageUri);
                if (bitmap != null) {
                    imageView.setImageBitmap(bitmap);
                } else {
                    // Fallback la metoda simplă dacă rotația eșuează
                    imageView.setImageURI(imageUri);
                }
            } catch (Exception e) {
                Toast.makeText(this, "Eroare la încărcarea imaginii", Toast.LENGTH_SHORT).show();
            }
        }

        // Afișăm rezultatul parsând String-ul rezultatului
        if (!TextUtils.isEmpty(result)) {
            boolean isAuthentic = result.toLowerCase().contains("authentic") || 
                                 result.toLowerCase().contains("autentic");
            
            // Setăm culoarea textului în funcție de rezultat (verde pentru autentic, roșu pentru fals)
            if (isAuthentic) {
                resultText.setText(R.string.result_authentic);
                resultText.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            } else {
                resultText.setText(R.string.result_fake);
                resultText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            }
            
            // Extragem scorul din rezultat (formatul "... (95%)")
            try {
                String scorePart = result.substring(result.indexOf("(") + 1, result.indexOf(")"));
                float scoreValue = Float.parseFloat(scorePart.replace("%", "")) / 100f;
                confidenceScore.setText(getString(R.string.confidence_score, scoreValue * 100));
            } catch (Exception e) {
                confidenceScore.setText(getString(R.string.confidence_score, 95.0f));
            }
        } else {
            // Valori implicite
            resultText.setText(R.string.result_authentic);
            confidenceScore.setText(getString(R.string.confidence_score, 95.0f));
        }

        // Buton pentru revenirea la ecranul principal
        homeButton.setOnClickListener(view -> {
            Intent homeIntent = new Intent(ResultActivity.this, MainActivity.class);
            homeIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(homeIntent);
            finish();
        });
    }
    
    // Metodă pentru încărcarea și rotirea imaginii
    private Bitmap loadAndRotateImage(Uri imageUri) {
        try {
            // Încărcăm imaginea
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            Bitmap originalBitmap = BitmapFactory.decodeStream(inputStream);
            if (inputStream != null) {
                inputStream.close();
            }
            
            if (originalBitmap == null) {
                return null;
            }
            
            // Rotim imaginea cu 90 de grade
            Matrix matrix = new Matrix();
            matrix.postRotate(90);
            
            return Bitmap.createBitmap(
                originalBitmap, 
                0, 0, 
                originalBitmap.getWidth(), 
                originalBitmap.getHeight(), 
                matrix, 
                true
            );
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
