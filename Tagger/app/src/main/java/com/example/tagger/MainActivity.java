package com.example.tagger;

import android.content.Intent;
import android.os.Bundle;
import android.os.Looper;
import android.view.View;
import android.widget.GridView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.os.HandlerCompat;
import android.os.Handler;

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
            R.drawable.louisvuitton, R.drawable.north, R.drawable.offwhite, R.drawable.ralphlauren,
            R.drawable.stussy, R.drawable.supreme, R.drawable.tommyhilfiger, R.drawable.versace
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Configuram GridView
        final GridView gridView = findViewById(R.id.brandGrid);
        
        // Optimizări pentru scroll
        gridView.setFastScrollEnabled(true);
        gridView.setScrollingCacheEnabled(false);
        gridView.setAnimationCacheEnabled(false);
        
        // Folosim HandlerCompat pentru a încărca adaptorul după ce UI-ul este pregătit
        // HandlerCompat oferă compatibilitate îmbunătățită pentru versiuni mai noi
        Handler mainHandler = HandlerCompat.createAsync(Looper.getMainLooper());
        mainHandler.post(() -> {
            BrandAdapter adapter = new BrandAdapter(this, brandNames, brandLogos);
            gridView.setAdapter(adapter);

            gridView.setOnItemClickListener((parent, view, position, id) -> {
                Intent intent = new Intent(MainActivity.this, ScanActivity.class);
                intent.putExtra("BRAND_NAME", brandNames[position]);
                startActivity(intent);
            });
            
            // Ascundem loading indicator dacă există
            View loadingIndicator = findViewById(R.id.loadingIndicator);
            if (loadingIndicator != null) {
                loadingIndicator.setVisibility(View.GONE);
            }
        });
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        // Forțăm redraw-ul GridView pentru a rezolva probleme de scrolling
        GridView gridView = findViewById(R.id.brandGrid);
        if (gridView != null && gridView.getAdapter() != null) {
            gridView.invalidateViews();
        }
    }
}