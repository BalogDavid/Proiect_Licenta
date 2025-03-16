package com.example.tagger

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.tagger.ui.theme.TaggerTheme

class ResultActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val brand = intent.getStringExtra("BRAND_NAME") ?: "Unknown"
        val result = intent.getStringExtra("RESULT") ?: "Necunoscut"

        setContent {
            TaggerTheme {
                ResultScreen(brand, result, onBackToHome = {
                    val intent = Intent(this@ResultActivity, MainActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP
                    startActivity(intent)
                })
            }
        }
    }
}

@Composable
fun ResultScreen(brand: String, result: String, onBackToHome: () -> Unit) {
    val color = if (result == "Autentic") Color.Green else Color.Red

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Rezultatul scanării pentru $brand",
            style = MaterialTheme.typography.headlineMedium
        )
        Spacer(modifier = Modifier.height(20.dp))
        Card(
            modifier = Modifier.padding(16.dp),
            colors = CardDefaults.cardColors(containerColor = color)
        ) {
            Text(
                text = result,
                style = MaterialTheme.typography.headlineLarge,
                color = Color.White,
                modifier = Modifier.padding(16.dp)
            )
        }
        Spacer(modifier = Modifier.height(20.dp))
        Button(onClick = onBackToHome) {
            Text("Înapoi la ecranul principal")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ResultScreenPreview() {
    TaggerTheme {
        ResultScreen("Louis Vuitton", "Autentic", {})
    }
}
