# Ghid de implementare Android pentru Detector Etichete

Acest ghid explică cum să integrezi modelul de detectare a etichetelor false/reale în aplicația Android.

## Specificații model

- **Format:** TensorFlow Lite (TFLite), compatibil cu TensorFlow 2.14
- **Dimensiune model:** ~3.2MB
- **Dimensiune input:** Imagine 128x128 pixeli, RGB, normalizată (0-1)
- **Output:** Valoare între 0 și 1
  - Valori > 0.75 indică o etichetă falsă (Fake)
  - Valori < 0.75 indică o etichetă autentică (Real)
- **Acuratețe:** ~77.78% pe setul de test

## Fișiere necesare

Pentru implementare ai nevoie de:

1. `model_etichete.tflite` - Modelul TFLite optimizat (3.2MB)
2. `etichete.txt` - Lista de etichete ("0 Fake Labels", "1 Authentic Labels")

Ambele fișiere se găsesc în folderul `model_android/`.

## Configurare proiect

### 1. Adaugă fișierele în proiect

Copiază fișierele în directorul `assets` al proiectului:

```
app/src/main/assets/
├── model_etichete.tflite
└── etichete.txt
```

### 2. Adaugă dependențele TensorFlow Lite

În fișierul `build.gradle` al modulului, adaugă:

```gradle
dependencies {
    // TensorFlow Lite pentru Android
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3'
}
```

## Implementare cod

### Java

```java
import android.content.Context;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.IOException;
import java.util.List;

public class DetectorEtichete {
    private static final String MODEL_PATH = "model_etichete.tflite";
    private static final String LABELS_PATH = "etichete.txt";
    private static final int IMAGE_SIZE = 128;
    private static final float THRESHOLD = 0.75f;
    
    private Interpreter interpreter;
    private List<String> labels;
    private Context context;
    
    public DetectorEtichete(Context context) throws IOException {
        this.context = context;
        this.interpreter = new Interpreter(loadModel());
        this.labels = loadLabels();
    }
    
    private MappedByteBuffer loadModel() throws IOException {
        // Încarcă modelul din assets
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    private List<String> loadLabels() throws IOException {
        // Încarcă etichetele din assets
        return FileUtil.loadLabels(context, LABELS_PATH);
    }
    
    public ResultatDetectie detecteazaEticheta(Bitmap bitmap) {
        // Redimensionează imaginea la dimensiunea așteptată
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        
        // Preprocesare imagine
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
        inputBuffer.order(ByteOrder.nativeOrder());
        
        int[] pixels = new int[IMAGE_SIZE * IMAGE_SIZE];
        resizedBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
        
        for (int pixel : pixels) {
            // Normalizare culori (0-1)
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            
            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }
        
        // Rulare inferență
        float[][] outputBuffer = new float[1][1];
        interpreter.run(inputBuffer, outputBuffer);
        
        // Procesare rezultat
        float probability = outputBuffer[0][0];
        boolean isFake = probability > THRESHOLD;
        
        int classIndex = isFake ? 0 : 1;
        String label = labels.get(classIndex);
        float confidence = isFake ? probability : 1 - probability;
        
        return new ResultatDetectie(isFake, probability, confidence, label);
    }
    
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
    
    public static class ResultatDetectie {
        private final boolean esteFalsa;
        private final float probabilitate;
        private final float incredere;
        private final String eticheta;
        
        public ResultatDetectie(boolean esteFalsa, float probabilitate, float incredere, String eticheta) {
            this.esteFalsa = esteFalsa;
            this.probabilitate = probabilitate;
            this.incredere = incredere;
            this.eticheta = eticheta;
        }
        
        public boolean esteFalsa() {
            return esteFalsa;
        }
        
        public float getProbabilitate() {
            return probabilitate;
        }
        
        public float getIncredere() {
            return incredere;
        }
        
        public String getEticheta() {
            return eticheta;
        }
    }
}
```

### Kotlin

```kotlin
import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class DetectorEtichete(private val context: Context) {
    private val MODEL_PATH = "model_etichete.tflite"
    private val LABELS_PATH = "etichete.txt"
    private val IMAGE_SIZE = 128
    private val THRESHOLD = 0.75f
    
    private var interpreter: Interpreter? = null
    private var etichete: List<String> = emptyList()
    
    init {
        // Inițializare model și etichete
        interpreter = Interpreter(incarcaModel())
        etichete = incarcaEtichete()
    }
    
    private fun incarcaModel(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    private fun incarcaEtichete(): List<String> {
        return context.assets.open(LABELS_PATH).bufferedReader().readLines()
    }
    
    fun detecteazaEticheta(bitmap: Bitmap): ResultatDetectie {
        // Redimensionează imaginea
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        
        // Preprocesare
        val inputBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resizedBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        
        for (pixel in pixels) {
            // Normalizare culori (0-1)
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        
        // Rulare model
        val outputBuffer = Array(1) { FloatArray(1) }
        inputBuffer.rewind()
        interpreter?.run(inputBuffer, outputBuffer)
        
        // Procesare rezultat
        val probabilitate = outputBuffer[0][0]
        val esteFalsa = probabilitate > THRESHOLD
        
        val indexClasa = if (esteFalsa) 0 else 1
        val eticheta = etichete[indexClasa]
        val incredere = if (esteFalsa) probabilitate else 1 - probabilitate
        
        return ResultatDetectie(esteFalsa, probabilitate, incredere, eticheta)
    }
    
    fun elibereaza() {
        interpreter?.close()
        interpreter = null
    }
    
    data class ResultatDetectie(
        val esteFalsa: Boolean,
        val probabilitate: Float,
        val incredere: Float,
        val eticheta: String
    )
}
```

## Exemplu de utilizare

```kotlin
// Inițializare
val detector = DetectorEtichete(context)

// Analizează o imagine
val bitmap = BitmapFactory.decodeFile("/storage/emulated/0/DCIM/Camera/eticheta.jpg")
val rezultat = detector.detecteazaEticheta(bitmap)

// Afișează rezultat
if (rezultat.esteFalsa) {
    // Etichetă falsă
    textView.text = "Etichetă FALSĂ detectată!"
    textView.setTextColor(Color.RED)
    statusTextView.text = "Încredere: ${rezultat.incredere * 100}%"
} else {
    // Etichetă autentică
    textView.text = "Etichetă AUTENTICĂ"
    textView.setTextColor(Color.GREEN)
    statusTextView.text = "Încredere: ${rezultat.incredere * 100}%"
}

// Eliberează resursele când nu mai sunt necesare
detector.elibereaza()
```

## Note importante

1. **Pragul de clasificare**: Modelul folosește un prag de 0.75 pentru clasificare:
   - Valori > 0.75 indică etichete false
   - Valori < 0.75 indică etichete autentice

2. **Performanță**: Modelul are o dimensiune de ~3.2MB, fiind optimizat pentru Android.

3. **Acuratețe**: Modelul are o acuratețe de ~77.78% pe setul de testare.

4. **Compatibilitate**: Testat și optimizat pentru TensorFlow Lite 2.14.0.

5. **Optimizări**: Pentru performanță mai bună:
   - Procesează imaginile în thread-uri separate
   - Folosește GPU delegate pentru inferență accelerată (unde este disponibil)

## Rezolvarea problemelor

### Eroare: "Cannot find TensorFlow Lite methods"
Asigură-te că utilizezi TensorFlow Lite 2.14.0 și că ai adăugat toate dependențele necesare.

### Eroare: "Model interpreter failed to load"
Verifică dacă fișierul model_etichete.tflite este corect amplasat în directorul assets și că este citit corect.

### Clasificare incorectă
Modelul are o acuratețe de ~77.78%. Pentru rezultate mai bune:
- Asigură-te că imaginile sunt bine iluminate
- Centrează eticheta în imagine
- Folosește un prag ajustat dacă este necesar (poți experimenta cu valori între 0.7 și 0.8) 