package com.example.tagger;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class ResultActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        ImageView imageView = findViewById(R.id.resultImage);
        TextView resultText = findViewById(R.id.resultText);
        TextView confidenceScore = findViewById(R.id.confidenceScore);
        Button homeButton = findViewById(R.id.homeButton);

        Intent intent = getIntent();
        String result = intent.getStringExtra("RESULT");
        float score = intent.getFloatExtra("CONFIDENCE", 0.0f);
        String imageUriString = intent.getStringExtra("IMAGE_URI");

        // Verifică dacă avem o imagine validă și o afișează
        if (imageUriString != null) {
            Uri imageUri = Uri.parse(imageUriString);
            imageView.setImageURI(imageUri);
        }

        // Afișează rezultatul și scorul
        resultText.setText(result != null ? result : "Rezultat necunoscut");
        confidenceScore.setText(String.format("Scor: %.2f%%", score * 100));

        // Buton pentru revenirea la ecranul principal
        homeButton.setOnClickListener(view -> {
            Intent homeIntent = new Intent(ResultActivity.this, MainActivity.class);
            homeIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(homeIntent);
            finish();
        });
    }
}
