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
            R.drawable.nike, R.drawable.balenciaga, R.drawable.bape, R.drawable.carhartt,
            R.drawable.chanel, R.drawable.gucci, R.drawable.jordan, R.drawable.levis,
            R.drawable.louisvuitton, R.drawable.thenorthface, R.drawable.offwhite, R.drawable.ralphlauren,
            R.drawable.stussy, R.drawable.supreme, R.drawable.tommyhilfiger, R.drawable.versace
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