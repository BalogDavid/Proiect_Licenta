package com.example.tagger

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.tagger.ui.theme.TaggerTheme
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ScanActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private var imageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val brand = intent.getStringExtra("BRAND_NAME") ?: "Unknown"

        setContent {
            TaggerTheme {
                ScanScreen(brand, onImageCaptured = { uri ->
                    imageUri = uri
                    processImageAndNavigate(brand, uri)
                })
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun processImageAndNavigate(brand: String, uri: Uri) {
        // TODO: Aici vom trimite imaginea către modelul ML pentru clasificare
        val result = "Autentic" // Placeholder
        val intent = Intent(this@ScanActivity, ResultActivity::class.java)
        intent.putExtra("BRAND_NAME", brand)
        intent.putExtra("RESULT", result)
        startActivity(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

@Composable
fun ScanScreen(brand: String, onImageCaptured: (Uri) -> Unit) {
    val context = LocalContext.current
    var imageCapture: ImageCapture? = remember { null }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Scanează eticheta pentru: $brand", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(20.dp))

        CameraPreview { capture ->
            imageCapture = capture
        }

        Spacer(modifier = Modifier.height(20.dp))
        Button(onClick = {
            imageCapture?.let { capture ->
                val photoFile = File(context.filesDir, "${System.currentTimeMillis()}.jpg")
                val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

                capture.takePicture(
                    outputOptions,
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageSavedCallback {
                        override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                            onImageCaptured(Uri.fromFile(photoFile))
                        }

                        override fun onError(exception: ImageCaptureException) {
                            Toast.makeText(context, "Eroare la capturare", Toast.LENGTH_SHORT).show()
                            Log.e("CameraX", "Eroare: ${exception.message}", exception)
                        }
                    }
                )
            }
        }) {
            Text("Capturează imaginea")
        }
    }
}

@Composable
fun CameraPreview(onCaptureReady: (ImageCapture) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    AndroidView(
        factory = { ctx ->
            val previewView = androidx.camera.view.PreviewView(ctx)
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageCapture = ImageCapture.Builder().build()
            onCaptureReady(imageCapture)

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("CameraX", "Eroare la inițializare cameră", e)
            }

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}

@Preview(showBackground = true)
@Composable
fun ScanScreenPreview() {
    TaggerTheme {
        ScanScreen("Louis Vuitton", onImageCaptured = {})
    }
}
