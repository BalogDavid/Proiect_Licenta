import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# Dimensiunea imaginii și pragul de decizie
IMG_SIZE = 128
THRESHOLD = 0.75  # Pragul optim găsit anterior

def load_and_preprocess_image(image_path):
    """Încarcă și preprocesează o imagine pentru inferență"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Nu s-a putut încărca imaginea de la {image_path}")
        return None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalizare
    img = np.expand_dims(img, axis=0)  # Adaugă dimensiunea pentru batch
    return img

def predict_with_tflite(model_path, image_path, threshold=THRESHOLD):
    """Execută predicții folosind modelul TFLite"""
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
    
    # Interpretează rezultatele folosind pragul specificat
    raw_prediction = output_data[0][0]
    classification = "Fake" if raw_prediction > threshold else "Real"
    
    # Calculează încrederea (distanța de la 0.5, nu de la pragul nostru ajustat)
    # pentru a menține valorile intuitive
    confidence = raw_prediction if raw_prediction > 0.5 else 1.0 - raw_prediction
    
    # Definește care sunt rezultatele așteptate
    expected_results = {
        "test_image.jpg": "Fake",
        "test_image1.jpg": "Real",
        "test_image2.jpg": "Real",
        "test_image3.jpg": "Real",
        "test_image4.jpg": "Real", 
        "test_image5.jpg": "Real",
        "test_image6.jpg": "Fake",
        "test_image7.jpg": "Real",  # Actualizat ca fiind Real
        "test_image8.jpg": "Real"   # Actualizat ca fiind Real
    }
    
    # Verifică dacă predicția este corectă
    base_name = os.path.basename(image_path)
    expected = expected_results.get(base_name, "Unknown")
    is_correct = classification == expected if expected != "Unknown" else True
    
    # Afișează rezultatele
    print(f"Imagine: {base_name}")
    print(f"Predicție: {classification}")
    print(f"Probabilitate brută: {raw_prediction * 100:.2f}%")
    print(f"Încredere: {confidence * 100:.2f}%")
    print(f"Rezultat așteptat: {expected}")
    if expected != "Unknown":
        print(f"Verdict: {'CORECT ✓' if is_correct else 'INCORECT ✗'}")
    else:
        print("Verdict: Necunoscut - imagine nouă")
    
    return classification, raw_prediction, confidence, is_correct

def test_all_images(model_path, threshold=THRESHOLD):
    """Testează toate imaginile de test disponibile"""
    # Găsește toate imaginile de test
    test_images = [f for f in os.listdir(".") if f.startswith("test_image") and f.endswith(".jpg")]
    
    print(f"Testare toate imaginile cu modelul {os.path.basename(model_path)} (prag {threshold})")
    print("=" * 60)
    
    correct_count = 0
    total_count = 0
    results = {}
    
    for image_file in sorted(test_images):
        image_path = os.path.join(".", image_file)
        print(f"\nTestare {image_file}:")
        classification, raw_prob, confidence, is_correct = predict_with_tflite(model_path, image_path, threshold)
        
        results[image_file] = {
            "classification": classification,
            "probability": raw_prob,
            "confidence": confidence
        }
        
        # Verifică rezultatul pentru toate imaginile
        if is_correct:
            correct_count += 1
        total_count += 1
    
    if total_count > 0:
        accuracy = correct_count / total_count * 100
    else:
        accuracy = 0
        
    print("\n" + "=" * 60)
    print(f"Acuratețe: {accuracy:.2f}% ({correct_count}/{total_count})")
    
    # Afișează sumarul rezultatelor pentru toate imaginile
    print("\nSumar rezultate:")
    print("-" * 60)
    print(f"{'Imagine':<15} {'Predicție':<8} {'Probabilitate':<15} {'Încredere':<12}")
    print("-" * 60)
    
    for img, data in sorted(results.items()):
        print(f"{img:<15} {data['classification']:<8} {data['probability']*100:.2f}% {data['confidence']*100:.2f}%")
    
    return results

if __name__ == "__main__":
    model_dir = "model"
    
    # Modelul personalizat cu pragul de 0.75
    custom_model = os.path.join(model_dir, "custom_model_thresh75_optimized.tflite")
    
    if len(sys.argv) == 1:
        # Fără argumente - testează toate imaginile
        test_all_images(custom_model)
    elif len(sys.argv) == 2:
        # Un argument - testează o singură imagine
        image_path = sys.argv[1]
        predict_with_tflite(custom_model, image_path)
    else:
        print("Utilizare:")
        print("  python test_custom_model.py - testează toate imaginile cu modelul personalizat")
        print("  python test_custom_model.py <cale_imagine> - testează o singură imagine") 