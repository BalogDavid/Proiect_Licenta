import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# Dimensiunea imaginii
IMG_SIZE = 128

# Prag ajustat pentru clasificare - valori mai mari = mai puține imagini clasificate ca Fake
# Setăm pragul mai mare pentru a reduce clasificările false pozitive
THRESHOLD = 0.7  # Ajustează această valoare pentru a schimba sensibilitatea - 0.5 este valoarea standard

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Nu s-a putut încărca imaginea de la {image_path}")
        return None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalizare
    img = np.expand_dims(img, axis=0)  # Adaugă dimensiunea pentru batch
    return img

def predict_with_tflite(model_path, image_path, threshold=THRESHOLD):
    # Încarcă modelul TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Obține detalii despre input și output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocesează imaginea
    input_data = load_and_preprocess_image(image_path)
    if input_data is None:
        return
    
    # Setează input-ul
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    # Rulează predicția
    interpreter.invoke()

    # Obține rezultatul
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    raw_prediction = output_data[0][0]
    
    # Aplică pragul ajustat
    result = "Fake" if raw_prediction > threshold else "Real"
    
    # Pentru a afișa încrederea, folosim distanța de la 0.5, nu de la pragul ajustat
    # (pentru a menține valorile de încredere intuitive)
    confidence = raw_prediction if raw_prediction > 0.5 else 1.0 - raw_prediction
    
    print(f"Rezultatul clasificării pentru {image_path}: {result}")
    print(f"Probabilitatea brută: {raw_prediction * 100:.2f}%")
    print(f"Încredere: {confidence * 100:.2f}%")
    print(f"Prag folosit: {threshold * 100:.2f}%")
    return result, raw_prediction, confidence

def test_all_images_with_threshold(threshold=THRESHOLD):
    """Testează toate imaginile de test folosind pragul specificat"""
    model_path = "model/custom_model_thresh75_optimized.tflite"  # Folosim modelul optimizat personalizat
    
    # Lista de imagini de test
    image_files = [f for f in os.listdir(".") if f.startswith("test_image") and f.endswith(".jpg")]
    
    print(f"Testare toate imaginile cu prag ajustat la {threshold * 100:.2f}%")
    print("=" * 60)
    
    results = {}
    
    for image_file in sorted(image_files):
        print(f"\nTestare {image_file}:")
        result, raw_prob, _ = predict_with_tflite(model_path, image_file, threshold)
        results[image_file] = {"result": result, "probability": raw_prob}
    
    print("\n" + "=" * 60)
    print("Sumar rezultate:")
    print("-" * 60)
    print(f"{'Imagine':<15} {'Rezultat':<10} {'Probabilitate':<15}")
    print("-" * 60)
    
    for img, data in sorted(results.items()):
        print(f"{img:<15} {data['result']:<10} {data['probability'] * 100:.2f}%")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Fără argumente - testează toate imaginile cu pragul implicit
        test_all_images_with_threshold()
    elif len(sys.argv) == 2:
        if sys.argv[1].replace('.', '', 1).isdigit():
            # Argument este un număr - folosește-l ca prag
            threshold = float(sys.argv[1])
            test_all_images_with_threshold(threshold)
        else:
            # Argument este o cale de imagine - testează doar acea imagine
            image_path = sys.argv[1]
            model_path = "model/custom_model_thresh75_optimized.tflite"
            predict_with_tflite(model_path, image_path)
    elif len(sys.argv) == 3:
        # Două argumente - model și imagine sau imagine și prag
        if sys.argv[2].replace('.', '', 1).isdigit():
            # Al doilea argument este un număr - imagine și prag
            image_path = sys.argv[1]
            threshold = float(sys.argv[2])
            model_path = "model/custom_model_thresh75_optimized.tflite"
            predict_with_tflite(model_path, image_path, threshold)
        else:
            # Model și imagine
            model_path = sys.argv[1]
            image_path = sys.argv[2]
            predict_with_tflite(model_path, image_path)
    else:
        print("Utilizare:")
        print("  python test_tflite_adjusted.py - testează toate imaginile cu pragul implicit")
        print("  python test_tflite_adjusted.py <prag> - testează toate imaginile cu pragul specificat (0.0-1.0)")
        print("  python test_tflite_adjusted.py <cale_imagine> - testează o imagine cu pragul implicit")
        print("  python test_tflite_adjusted.py <cale_imagine> <prag> - testează o imagine cu pragul specificat") 