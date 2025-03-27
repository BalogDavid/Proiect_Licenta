package com.example.tagger;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.media.ExifInterface;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class ResultActivity extends AppCompatActivity {
    private Bitmap resultBitmap;
    private Uri imageUri;
    private String brandName;
    private String classificationResult;
    
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
        Button saveResultButton = findViewById(R.id.saveResultButton);

        // Obținem datele din intent
        Intent intent = getIntent();
        brandName = intent.getStringExtra("BRAND_NAME");
        classificationResult = intent.getStringExtra("RESULT");
        String imageUriString = intent.getStringExtra("IMAGE_URI");

        // Setăm numele brandului
        if (!TextUtils.isEmpty(brandName)) {
            brandNameText.setText(brandName);
        }

        // Verificăm dacă avem o imagine validă și o afișează cu orientarea corectă
        if (imageUriString != null) {
            try {
                imageUri = Uri.parse(imageUriString);
                
                // Încărcăm imaginea și o rotim corect în funcție de metadatele EXIF
                resultBitmap = loadAndRotateImageCorrectly(imageUri);
                if (resultBitmap != null) {
                    imageView.setImageBitmap(resultBitmap);
                } else {
                    // Fallback la metoda simplă dacă rotația eșuează
                    imageView.setImageURI(imageUri);
                }
            } catch (Exception e) {
                Toast.makeText(this, "Eroare la încărcarea imaginii", Toast.LENGTH_SHORT).show();
            }
        }

        // Afișăm rezultatul parsând String-ul rezultatului
        if (!TextUtils.isEmpty(classificationResult)) {
            // Interpretăm formatul "X Label" cu valoare între 0-1
            boolean isAuthentic = false;
            float scoreValue = 0.5f; // Valoare implicită
            
            // Verificăm dacă rezultatul conține cuvântul "Authentic"
            if (classificationResult.toLowerCase().contains("authentic")) {
                isAuthentic = true;
                // Extrage scorul din rezultat - formatul tipic este "1 Authentic Labels (0.xxxxx)"
                try {
                    String[] parts = classificationResult.split("\\(");
                    if (parts.length > 1) {
                        String scorePart = parts[1].replace(")", "").trim();
                        scoreValue = Float.parseFloat(scorePart);
                    }
                } catch (Exception e) {
                    // Folosim scorul implicit
                }
            } else if (classificationResult.toLowerCase().contains("fake")) {
                isAuthentic = false;
                // Extrage scorul din rezultat - formatul tipic este "0 Fake Labels (0.xxxxx)"
                try {
                    String[] parts = classificationResult.split("\\(");
                    if (parts.length > 1) {
                        String scorePart = parts[1].replace(")", "").trim();
                        scoreValue = Float.parseFloat(scorePart);
                        // Pentru etichete false, inversăm scorul pentru a arăta "cât de sigur este că e fals"
                        scoreValue = 1.0f - scoreValue;
                    }
                } catch (Exception e) {
                    // Folosim scorul implicit
                }
            }
            
            // Setăm culoarea textului în funcție de rezultat (verde pentru autentic, roșu pentru fals)
            if (isAuthentic) {
                resultText.setText(R.string.result_authentic);
                resultText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_dark));
            } else {
                resultText.setText(R.string.result_fake);
                resultText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark));
            }
            
            // Afișăm scorul de încredere ca procentaj
            confidenceScore.setText(getString(R.string.confidence_score, scoreValue * 100));
            
        } else {
            // Valori implicite
            resultText.setText(R.string.result_inconclusive);
            resultText.setTextColor(ContextCompat.getColor(this, android.R.color.darker_gray));
            confidenceScore.setText(getString(R.string.confidence_score, 50.0f));
        }

        // Buton pentru revenirea la ecranul principal
        homeButton.setOnClickListener(view -> {
            Intent homeIntent = new Intent(ResultActivity.this, MainActivity.class);
            homeIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(homeIntent);
            finish();
        });
        
        // Buton pentru salvarea rezultatului
        saveResultButton.setOnClickListener(view -> {
            showSaveOptionsDialog();
        });
    }
    
    // Metodă pentru afișarea dialogului de salvare
    private void showSaveOptionsDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Salvare imagine");
        
        // Opțiunile pentru salvare
        String[] options = {
                getString(R.string.save_to_gallery),
                getString(R.string.save_to_files)
        };
        
        builder.setItems(options, (dialog, which) -> {
            switch (which) {
                case 0: // Salvare în galerie
                    saveResultToGallery();
                    break;
                case 1: // Salvare în fișiere
                    saveResultToFiles();
                    break;
            }
        });
        
        builder.show();
    }
    
    // Salvează imaginea rezultat în galerie
    private void saveResultToGallery() {
        Bitmap resultImageWithText = createResultImage();
        if (resultImageWithText != null) {
            try {
                String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
                String title = "Tagger_" + timeStamp;
                String description = "Tagger result for " + brandName + ": " + classificationResult;
                
                // Utilizăm MediaStore API modern pentru Android 10+ (Q)
                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.DISPLAY_NAME, title);
                values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                values.put(MediaStore.Images.Media.DESCRIPTION, description);
                values.put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000);
                values.put(MediaStore.Images.Media.DATE_TAKEN, System.currentTimeMillis());
                
                Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (uri != null) {
                    try (OutputStream outputStream = getContentResolver().openOutputStream(uri)) {
                        if (outputStream != null) {
                            resultImageWithText.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
                        }
                    }
                    Toast.makeText(this, "Imagine salvată în galerie", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Toast.makeText(this, "Eroare la salvarea imaginii: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    // Salvează imaginea rezultat în directorul de fișiere al aplicației
    private void saveResultToFiles() {
        Bitmap resultImageWithText = createResultImage();
        if (resultImageWithText != null) {
            try {
                File pictureFile = createImageFile();
                FileOutputStream fos = new FileOutputStream(pictureFile);
                resultImageWithText.compress(Bitmap.CompressFormat.JPEG, 100, fos);
                fos.close();
                
                // Utilizăm API MediaScanner modern în loc de ACTION_MEDIA_SCANNER_SCAN_FILE deprecated
                MediaScannerConnection.scanFile(
                    this,
                    new String[] { pictureFile.getAbsolutePath() },
                    new String[] { "image/jpeg" },
                    null
                );
                
                Toast.makeText(this, "Imagine salvată în: " + pictureFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
            } catch (Exception e) {
                Toast.makeText(this, "Eroare la salvarea imaginii: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    // Creează un fișier pentru salvarea imaginii
    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "Tagger_" + timeStamp;
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        return File.createTempFile(imageFileName, ".jpg", storageDir);
    }
    
    // Creează o imagine cu rezultatul suprapus
    private Bitmap createResultImage() {
        if (resultBitmap == null) return null;
        
        // Creăm o copie a imaginii originale
        Bitmap resultImage = resultBitmap.copy(resultBitmap.getConfig(), true);
        Canvas canvas = new Canvas(resultImage);
        
        // Configurăm textul pentru rezultat
        Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setTextSize(50);
        paint.setTypeface(Typeface.DEFAULT_BOLD);
        paint.setShadowLayer(5, 0, 0, Color.BLACK);
        
        // Determinăm culoarea fundalului în funcție de rezultat
        int backgroundColor;
        String result;
        
        if (classificationResult.toLowerCase().contains("authentic")) {
            backgroundColor = Color.argb(180, 0, 200, 0);
            result = getString(R.string.result_authentic);
        } else if (classificationResult.toLowerCase().contains("fake")) {
            backgroundColor = Color.argb(180, 200, 0, 0);
            result = getString(R.string.result_fake);
        } else {
            backgroundColor = Color.argb(180, 150, 150, 150);
            result = getString(R.string.result_inconclusive);
        }
        
        // Creăm un fundal pentru text
        Paint bgPaint = new Paint();
        bgPaint.setColor(backgroundColor);
        bgPaint.setStyle(Paint.Style.FILL);
        
        // Extragem scorul dacă există
        String scoreText = "";
        try {
            String[] parts = classificationResult.split("\\(");
            if (parts.length > 1) {
                String scorePart = parts[1].replace(")", "").trim();
                float scoreValue = Float.parseFloat(scorePart);
                if (result.equals(getString(R.string.result_fake))) {
                    scoreValue = 1.0f - scoreValue;
                }
                scoreText = "Scor: " + String.format(Locale.getDefault(), "%.2f%%", scoreValue * 100);
            }
        } catch (Exception e) {
            // Ignorăm eroarea, nu afișăm scorul
        }
        
        // Calculăm dimensiunile textului
        Rect bounds = new Rect();
        paint.getTextBounds(result, 0, result.length(), bounds);
        
        // Poziția casetei de text - acum în stânga jos
        int padding = 15;
        int left = padding;
        
        // Poziționăm textul în stânga jos
        int textWidth = Math.max(bounds.width() + padding * 2, 
                (!scoreText.isEmpty()) ? (int) paint.measureText(scoreText) + padding * 2 : 0);
        int right = Math.min(
            left + textWidth,
            (int)(resultImage.getWidth() * 0.35f)
        );
        
        int height = bounds.height() + padding * 2 + (scoreText.isEmpty() ? 0 : 40);
        
        // Calculăm poziția y pentru a plasa textul în partea de jos
        int bottom = resultImage.getHeight() - padding;
        int top = bottom - height;
        
        // Desenăm fundalul pentru text
        canvas.drawRect(left, top, right, bottom, bgPaint);
        
        // Desenăm textul rezultatului
        canvas.drawText(result, left + padding, top + bounds.height() + padding / 2, paint);
        
        // Desenăm textul scorului dacă există
        if (!scoreText.isEmpty()) {
            Paint scorePaint = new Paint();
            scorePaint.setColor(Color.WHITE);
            scorePaint.setTextSize(40);
            scorePaint.setShadowLayer(3, 0, 0, Color.BLACK);
            
            canvas.drawText(scoreText, left + padding, top + bounds.height() + padding + 25, scorePaint);
        }
        
        return resultImage;
    }
    
    // Metodă pentru determinarea orientării corecte a imaginii din metadatele EXIF
    private Bitmap loadAndRotateImageCorrectly(Uri imageUri) {
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
            
            // Obținem orientarea din metadatele EXIF dacă există
            int rotation = 0;
            try {
                if (imageUri.getScheme().equals("content")) {
                    inputStream = getContentResolver().openInputStream(imageUri);
                    if (inputStream != null) {
                        ExifInterface exif = new ExifInterface(inputStream);
                        int orientation = exif.getAttributeInt(
                                ExifInterface.TAG_ORIENTATION,
                                ExifInterface.ORIENTATION_NORMAL);
                        
                        switch (orientation) {
                            case ExifInterface.ORIENTATION_ROTATE_90:
                                rotation = 90;
                                break;
                            case ExifInterface.ORIENTATION_ROTATE_180:
                                rotation = 180;
                                break;
                            case ExifInterface.ORIENTATION_ROTATE_270:
                                rotation = 270;
                                break;
                        }
                        inputStream.close();
                    }
                } else if (imageUri.getScheme().equals("file")) {
                    ExifInterface exif = new ExifInterface(imageUri.getPath());
                    int orientation = exif.getAttributeInt(
                            ExifInterface.TAG_ORIENTATION,
                            ExifInterface.ORIENTATION_NORMAL);
                    
                    switch (orientation) {
                        case ExifInterface.ORIENTATION_ROTATE_90:
                            rotation = 90;
                            break;
                        case ExifInterface.ORIENTATION_ROTATE_180:
                            rotation = 180;
                            break;
                        case ExifInterface.ORIENTATION_ROTATE_270:
                            rotation = 270;
                            break;
                    }
                }
            } catch (Exception e) {
                // Încearcă să rotească 90 de grade deoarece imaginile din galerie sunt de obicei rotite
                rotation = 90;
            }
            
            // Rotim imaginea cu unghiul necesar
            if (rotation != 0) {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotation);
                
                return Bitmap.createBitmap(
                    originalBitmap, 
                    0, 0, 
                    originalBitmap.getWidth(), 
                    originalBitmap.getHeight(), 
                    matrix, 
                    true
                );
            }
            
            return originalBitmap;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
