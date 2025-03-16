package com.example.tagger

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.tagger.ui.theme.TaggerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            TaggerTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    BrandList(
                        brands = listOf("Louis Vuitton", "Gucci", "Prada", "Balenciaga", "Versace"),
                        onBrandClick = { brand ->
                            val intent = Intent(this, ScanActivity::class.java)
                            intent.putExtra("BRAND_NAME", brand)
                            startActivity(intent)
                        },
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun BrandList(brands: List<String>, onBrandClick: (String) -> Unit, modifier: Modifier = Modifier) {
    LazyColumn(modifier = modifier.padding(16.dp)) {
        items(brands) { brand ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
                    .clickable { onBrandClick(brand) },
                elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(text = brand, style = MaterialTheme.typography.bodyLarge)
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun BrandListPreview() {
    TaggerTheme {
        BrandList(
            brands = listOf("Louis Vuitton", "Gucci", "Prada", "Balenciaga", "Versace"),
            onBrandClick = {}
        )
    }
}
