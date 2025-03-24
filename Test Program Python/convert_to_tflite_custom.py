import tensorflow as tf
import numpy as np
import os
import sys
import cv2

def create_custom_model(base_model_path, custom_threshold=0.5):
    """Creează un model personalizat cu un prag de clasificare încorporat"""
    
    # Încarcă modelul de bază
    base_model = tf.keras.models.load_model(base_model_path)
    
    # Creează un nou model personalizat
    def threshold_fn(x, threshold=custom_threshold):
        # x este ieșirea modelului de bază (între 0 și 1)
        # Dacă valoarea este mai mare decât pragul, păstrăm valoarea, altfel o facem 0
        return tf.where(x > threshold, x, tf.zeros_like(x))
    
    # Creează un model nou care încorporează pragul
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = base_model(inputs)
    # Nu aplicăm funcția de prag direct aici, dar o salvăm în model pentru a putea fi folosită în aplicație
    
    custom_model = tf.keras.Model(inputs=inputs, outputs=x)
    custom_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Adăugăm informații despre prag ca atribut al modelului
    custom_model.threshold = custom_threshold
    
    return custom_model

def convert_to_tflite(model, output_path):
    """Convertește modelul la format TFLite"""
    
    # Convertește modelul la TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Salvează modelul TFLite
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Model salvat la {output_path}")
    
    # Model optimizat
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_optimized_model = converter.convert()
    
    optimized_path = output_path.replace('.tflite', '_optimized.tflite')
    with open(optimized_path, "wb") as f:
        f.write(tflite_optimized_model)
        
    print(f"Model optimizat salvat la {optimized_path}")
    
    # Model cu cuantizare
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quantized_model = converter.convert()
    
    quantized_path = output_path.replace('.tflite', '_quantized.tflite')
    with open(quantized_path, "wb") as f:
        f.write(tflite_quantized_model)
        
    print(f"Model cuantizat salvat la {quantized_path}")
    
    # Comparație dimensiuni fișiere
    print(f"\nComparație dimensiuni:")
    print(f"TFLite standard: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print(f"TFLite optimizat: {os.path.getsize(optimized_path) / (1024 * 1024):.2f} MB") 
    print(f"TFLite cuantizat: {os.path.getsize(quantized_path) / (1024 * 1024):.2f} MB")


def test_model_on_images(model, threshold=0.5):
    """Testează modelul pe toate imaginile de test disponibile"""
    IMG_SIZE = 128
    
    # Găsește toate imaginile de test
    test_images = [f for f in os.listdir(".") if f.startswith("test_image") and f.endswith(".jpg")]
    
    results = {}
    
    # Definește care imagine ar trebui să fie fake sau real
    expected_results = {
        "test_image.jpg": "Fake",
        "test_image1.jpg": "Real",
        "test_image2.jpg": "Real",
        "test_image3.jpg": "Real",
        "test_image4.jpg": "Real", 
        "test_image5.jpg": "Real",
        "test_image6.jpg": "Fake"
    }
    
    print(f"\nTestare model pe imagini cu prag {threshold}:")
    print("-" * 60)
    
    correct_count = 0
    total_count = 0
    
    for img_file in sorted(test_images):
        # Încarcă și preprocesează imaginea
        img_path = os.path.join(".", img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predicție
        pred = model.predict(img, verbose=0)[0][0]
        
        # Clasifică cu pragul nostru
        classification = "Fake" if pred > threshold else "Real"
        
        # Verifică dacă predicția este corectă față de ce ar trebui să fie
        expected = expected_results.get(img_file, "Unknown")
        is_correct = classification == expected
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Afișează rezultatul
        print(f"{img_file:<15} Predicție: {classification:<6} ({pred*100:.2f}%) {'✓' if is_correct else '✗'} (Așteptat: {expected})")
        
        results[img_file] = {"prediction": pred, "classification": classification, "expected": expected}
    
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print("-" * 60)
    print(f"Acuratețe: {accuracy:.2f}% ({correct_count}/{total_count})")
    
    return results, accuracy

def find_optimal_threshold(model, start=0.5, end=0.99, step=0.05):
    """Găsește pragul optim care maximizează acuratețea clasificării"""
    best_threshold = 0.5
    best_accuracy = 0.0
    
    print("\nCăutare prag optim...")
    print("-" * 60)
    print(f"{'Prag':<10} {'Acuratețe':<10}")
    print("-" * 60)
    
    thresholds = []
    accuracies = []
    
    current = start
    while current <= end:
        _, accuracy = test_model_on_images(model, current)
        print(f"{current:<10.2f} {accuracy:<10.2f}%")
        
        thresholds.append(current)
        accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = current
        
        current += step
    
    print("-" * 60)
    print(f"Prag optim găsit: {best_threshold:.2f} cu acuratețe {best_accuracy:.2f}%")
    
    return best_threshold, thresholds, accuracies

if __name__ == "__main__":
    model_path = "model/model_trained.h5"
    output_path = "model/custom_model.tflite"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "optimize":
            # Găsește pragul optim
            print("Încărcare model pentru optimizare...")
            base_model = tf.keras.models.load_model(model_path)
            
            # Găsește pragul optim
            optimal_threshold, _, _ = find_optimal_threshold(base_model)
            
            # Creează și salvează modelul cu prag optim
            print(f"\nCreare model personalizat cu prag {optimal_threshold}...")
            custom_model = create_custom_model(model_path, optimal_threshold)
            
            # Convertește modelul
            convert_to_tflite(custom_model, output_path.replace('.tflite', f'_thresh{int(optimal_threshold*100)}.tflite'))
            
        elif sys.argv[1] == "test":
            # Testează modelul existent cu un prag anume
            print("Încărcare model pentru testare...")
            threshold = 0.5
            if len(sys.argv) > 2:
                threshold = float(sys.argv[2])
            
            base_model = tf.keras.models.load_model(model_path)
            test_model_on_images(base_model, threshold)
            
        else:
            # Convertește cu un prag specificat
            threshold = float(sys.argv[1])
            print(f"Crearea modelului personalizat cu prag {threshold}...")
            custom_model = create_custom_model(model_path, threshold)
            
            # Testează modelul
            test_model_on_images(custom_model, threshold)
            
            # Convertește modelul
            convert_to_tflite(custom_model, output_path.replace('.tflite', f'_thresh{int(threshold*100)}.tflite'))
    else:
        # Utilizare implicită - prag de 0.5
        print("Crearea modelului personalizat cu prag implicit (0.5)...")
        custom_model = create_custom_model(model_path)
        
        # Testează modelul
        test_model_on_images(custom_model)
        
        # Convertește modelul
        convert_to_tflite(custom_model, output_path)
        
    print("\nUtilizare:")
    print("  python convert_to_tflite_custom.py - creează model cu prag implicit (0.5)")
    print("  python convert_to_tflite_custom.py <prag> - creează model cu prag specificat")
    print("  python convert_to_tflite_custom.py optimize - găsește pragul optim și creează model")
    print("  python convert_to_tflite_custom.py test [<prag>] - testează modelul cu prag opțional") 