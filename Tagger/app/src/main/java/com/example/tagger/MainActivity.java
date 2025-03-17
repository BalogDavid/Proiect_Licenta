package com.example.tagger;

import android.os.Bundle;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;


import android.content.Intent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.GridView;


public class MainActivity extends AppCompatActivity {

    private String[] brandNames = {
            "Nike", "Balenciaga", "Bape", "Carhartt",
            "Chanel", "Gucci", "Jordan", "Levi's",
            "Louis Vuitton", "The North Face", "Off-White", "Ralph Lauren",
            "Stussy", "Supreme", "Tommy Hilfiger", "Versace"
    };

    private int[] brandLogos = {
            R.drawable.Nike, R.drawable.Balenciaga, R.drawable.Bape, R.drawable.Carhartt,
            R.drawable.Chanel, R.drawable.Gucci, R.drawable.Jordan, R.drawable.Levis,
            R.drawable.LouisVuitton, R.drawable.North, R.drawable.OffWhite, R.drawable.RalphLauren,
            R.drawable.Stussy, R.drawable.Supreme, R.drawable.TommyHilfiger, R.drawable.Versace
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        GridView gridView = findViewById(R.id.brandGrid);
        BrandAdapter adapter = new BrandAdapter(this, brandNames, brandLogos);
        gridView.setAdapter(adapter);

        gridView.setOnItemClickListener((parent, view, position, id) -> {
            Intent intent = new Intent(MainActivity.this, ScanActivity.class);
            intent.putExtra("BRAND_NAME", brandNames[position]);
            startActivity(intent);
        });
    }
}