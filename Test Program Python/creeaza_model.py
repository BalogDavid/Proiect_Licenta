import tensorflow as tf
import numpy as np
import os
import cv2
import argparse

# Afișăm un mesaj despre versiunea TensorFlow
print(f"Folosesc TensorFlow {tf.__version__} pentru crearea modelului TFLite")

def creeaza_model_nou():
    """
    Creează un model CNN simplu pentru clasificarea etichetelor
    
    Returnează:
    - Modelul compilat
    """
    model = tf.keras.Sequential([
        # Prima convoluție
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # A doua convoluție
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # A treia convoluție
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilare model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model creat cu succces")
    model.summary()
    
    return model

def creeaza_model_tflite(model, model_tflite, prag=0.75):
    """
    Convertește modelul în format TFLite
    
    Parametri:
    - model: Modelul Keras
    - model_tflite: Calea unde va fi salvat modelul TFLite
    - prag: Pragul pentru clasificare (implicit 0.75)
    """
    print(f"Convertesc modelul la TFLite cu prag {prag}...")
    
    # Convertește la TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Setează opțiunile compatibile cu TensorFlow 2.14
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Optimizează modelul pentru dimensiune și viteză
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Aceste opțiuni sunt recomandate pentru TensorFlow 2.14
    converter.experimental_new_converter = True
    
    # Convertește modelul
    tflite_model = converter.convert()
    
    # Creăm directorul dacă nu există
    os.makedirs(os.path.dirname(model_tflite), exist_ok=True)
    
    # Salvează modelul
    with open(model_tflite, "wb") as f:
        f.write(tflite_model)
    
    dimensiune = os.path.getsize(model_tflite) / (1024 * 1024)
    print(f"Model salvat: {model_tflite} ({dimensiune:.2f} MB)")
    
    return model_tflite

def creeaza_fisier_etichete(cale):
    """
    Creează fișierul de etichete pentru Android
    
    Parametri:
    - cale: Calea unde va fi salvat fișierul de etichete
    """
    # Creăm directorul dacă nu există
    os.makedirs(os.path.dirname(cale), exist_ok=True)
    
    with open(cale, "w") as f:
        f.write("0 Fake Labels\n1 Authentic Labels")
    
    print(f"Fișier de etichete creat: {cale}")

if __name__ == "__main__":
    # Definește argumentele
    parser = argparse.ArgumentParser(description="Crează model TFLite pentru detectarea etichetelor fake/real")
    parser.add_argument("--prag", type=float, default=0.75, help="Pragul pentru clasificare (implicit: 0.75)")
    args = parser.parse_args()
    
    print(f"Creez un model nou compatibil cu TensorFlow {tf.__version__}")
    
    # Creăm modelul
    model = creeaza_model_nou()
    
    # Creează model TFLite
    model_tflite = "model/model_etichete.tflite"
    creeaza_model_tflite(model, model_tflite, args.prag)
    
    # Creează directoriul pentru Android dacă nu există
    if not os.path.exists("model_android"):
        os.makedirs("model_android")
        
    # Copiază modelul în folderul Android
    import shutil
    android_model = "model_android/model_etichete.tflite"
    shutil.copy(model_tflite, android_model)
    print(f"Model copiat pentru Android: {android_model}")
    
    # Creează fișierul de etichete
    etichete = "model_android/etichete.txt"
    creeaza_fisier_etichete(etichete)
    
    print("\nProcesul a fost finalizat cu succes!")
    print("Un model nou a fost creat și salvat în formatele necesare.")
    print("Utilizare model în Android: Consultați fișierul 'ghid_android.md'") 