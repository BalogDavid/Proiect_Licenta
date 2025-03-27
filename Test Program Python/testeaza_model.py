import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

# Afișăm un mesaj despre versiunea TensorFlow
print(f"Folosesc TensorFlow pentru testarea modelului TFLite")

def pregateste_imagine(cale_imagine, dimensiune=128):
    """
    Pregătește o imagine pentru analiza cu modelul
    
    Parametri:
    - cale_imagine: Calea către imaginea de testat
    - dimensiune: Dimensiunea la care să fie redimensionată imaginea
    
    Returnează:
    - Imaginea pregătită pentru model
    """
    # Încarcă imaginea
    img = cv2.imread(cale_imagine)
    if img is None:
        raise ValueError(f"Nu s-a putut încărca imaginea: {cale_imagine}")
    
    # Redimensionează
    img = cv2.resize(img, (dimensiune, dimensiune))
    
    # Normalizează (0-1)
    img = img.astype('float32') / 255.0
    
    # Adaugă dimensiune batch
    img = np.expand_dims(img, axis=0)
    
    return img

def testeaza_imagine_tflite(model_tflite, cale_imagine, prag=0.5):
    """
    Testează o imagine cu modelul TFLite
    
    Parametri:
    - model_tflite: Calea către modelul TFLite
    - cale_imagine: Calea către imaginea de testat
    - prag: Pragul pentru clasificare (implicit 0.5)
    
    Returnează:
    - eticheta: "Fake" sau "Real"
    - eticheta_android: "Fake Labels" sau "Authentic Labels"
    - probabilitate: Valoarea brută a predicției
    - incredere: Nivelul de încredere în predicție
    """
    # Verifică dacă modelul există
    if not os.path.exists(model_tflite):
        raise FileNotFoundError(f"Modelul {model_tflite} nu a fost găsit!")
    
    # Pregătește imaginea
    img = pregateste_imagine(cale_imagine)
    
    # Încarcă modelul TFLite folosind opțiuni compatibile cu TF 2.14
    try:
        interpreter = tf.lite.Interpreter(model_path=model_tflite)
        interpreter.allocate_tensors()
        
        # Obține detaliile de input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Setează input-ul
        interpreter.set_tensor(input_details[0]['index'], img)
        
        # Rulează modelul
        interpreter.invoke()
        
        # Obține output-ul
        output = interpreter.get_tensor(output_details[0]['index'])
        probabilitate = output[0][0]
    except Exception as e:
        print(f"Eroare la rularea modelului: {e}")
        raise
    
    # Clasifică rezultatul
    if probabilitate > prag:
        eticheta = "Fake"
        eticheta_android = "Fake Labels"
        incredere = probabilitate
    else:
        eticheta = "Real"
        eticheta_android = "Authentic Labels"
        incredere = 1 - probabilitate
    
    return eticheta, eticheta_android, probabilitate, incredere

def testeaza_toate_imaginile(model_tflite, prag=0.5):
    """
    Testează toate imaginile din directorul curent
    
    Parametri:
    - model_tflite: Calea către modelul TFLite
    - prag: Pragul pentru clasificare (implicit 0.5)
    """
    # Rezultate așteptate
    rezultate_asteptate = {
        "test_image.jpg": "Fake",
        "test_image1.jpg": "Real",
        "test_image2.jpg": "Real",
        "test_image3.jpg": "Real",
        "test_image4.jpg": "Real", 
        "test_image5.jpg": "Real",
        "test_image6.jpg": "Fake",
        "test_image7.jpg": "Real",
        "test_image8.jpg": "Real"
    }
    
    # Găsește toate imaginile de test
    imagini_test = [f for f in os.listdir(".") if f.startswith("test_image") and f.endswith(".jpg")]
    corecte = 0
    total = 0
    rezultate = {}
    
    print(f"\nTestare model: {os.path.basename(model_tflite)} (prag {prag})")
    print("=" * 50)
    
    for imagine in sorted(imagini_test):
        cale_imagine = os.path.join(".", imagine)
        print(f"\nTestare {imagine}:")
        
        try:
            # Testează imaginea
            eticheta, eticheta_android, prob, incredere = testeaza_imagine_tflite(model_tflite, cale_imagine, prag)
            
            # Verifică corectitudinea
            asteptat = rezultate_asteptate.get(imagine, "Necunoscut")
            corect = eticheta == asteptat
            
            # Salvează rezultatul
            rezultate[imagine] = {
                "eticheta": eticheta,
                "eticheta_android": eticheta_android,
                "probabilitate": prob,
                "incredere": incredere
            }
            
            # Afișează rezultate
            print(f"Imagine: {imagine}")
            print(f"Predicție: {eticheta} ({eticheta_android})")
            print(f"Probabilitate: {prob*100:.2f}%")
            print(f"Încredere: {incredere*100:.2f}%")
            print(f"Rezultat așteptat: {asteptat}")
            print(f"Verdict: {'CORECT ✓' if corect else 'INCORECT ✗'}")
            
            if corect:
                corecte += 1
            total += 1
                
        except Exception as e:
            print(f"Eroare la testarea imaginii {imagine}: {e}")
    
    # Calculează acuratețea
    acuratete = corecte / total * 100 if total > 0 else 0
    print("\n" + "=" * 50)
    print(f"Acuratețe: {acuratete:.2f}% ({corecte}/{total})")
    
    # Afișează sumarul
    print("\nSumar rezultate:")
    print("-" * 60)
    print(f"{'Imagine':<15} {'Eticheta':<10} {'Eticheta Android':<20} {'Probabilitate':<15} {'Încredere'}")
    print("-" * 60)
    
    for imagine in sorted(rezultate.keys()):
        r = rezultate[imagine]
        print(f"{imagine:<15} {r['eticheta']:<10} {r['eticheta_android']:<20} {r['probabilitate']*100:.2f}% {r['incredere']*100:.2f}%")
    
    return acuratete, rezultate

if __name__ == "__main__":
    # Definește argumentele
    parser = argparse.ArgumentParser(description="Testează modelul TFLite de detectare a etichetelor fake/real")
    parser.add_argument("--model", type=str, default="model_android/model_etichete.tflite", help="Calea către modelul TFLite")
    parser.add_argument("--prag", type=float, default=0.5, help="Pragul pentru clasificare (implicit: 0.5)")
    parser.add_argument("--imagine", type=str, help="Testează o singură imagine (opțional)")
    args = parser.parse_args()
    
    # Verifică dacă modelul există
    if not os.path.exists(args.model):
        print(f"EROARE: Modelul {args.model} nu a fost găsit!")
        print("Rulați mai întâi 'python creeaza_model.py' pentru a crea modelul")
        exit(1)
    
    if args.imagine:
        # Verifică dacă imaginea există
        if not os.path.exists(args.imagine):
            print(f"EROARE: Imaginea {args.imagine} nu a fost găsită!")
            exit(1)
            
        # Testează o singură imagine
        print(f"Testez imaginea {args.imagine} cu modelul {args.model}")
        try:
            eticheta, eticheta_android, prob, incredere = testeaza_imagine_tflite(args.model, args.imagine, args.prag)
            print(f"Rezultat: {eticheta} ({eticheta_android})")
            print(f"Probabilitate: {prob*100:.2f}%")
            print(f"Încredere: {incredere*100:.2f}%")
        except Exception as e:
            print(f"Eroare: {e}")
    else:
        # Testează toate imaginile
        testeaza_toate_imaginile(args.model, args.prag) 