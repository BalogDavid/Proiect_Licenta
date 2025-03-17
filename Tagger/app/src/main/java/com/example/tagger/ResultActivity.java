package com.example.tagger;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class ResultActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        TextView resultText = findViewById(R.id.resultText);
        TextView confidenceScore = findViewById(R.id.confidenceScore);

        Intent intent = getIntent();
        String result = intent.getStringExtra("RESULT");
        float score = intent.getFloatExtra("CONFIDENCE", 0.0f);

        resultText.setText(result);
        confidenceScore.setText("Scor: " + String.format("%.2f", score * 100) + "%");
    }
}
