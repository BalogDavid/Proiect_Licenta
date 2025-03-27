import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Afișăm un mesaj despre versiunea TensorFlow
print(f"Folosesc TensorFlow {tf.__version__} pentru crearea modelului TFLite")

def creeaza_model_nou(learning_rate=0.001):
    """
    Creează un model CNN îmbunătățit pentru clasificarea etichetelor cu mai multă regularizare
    
    Parametri:
    - learning_rate: Rata de învățare pentru optimizer
    
    Returnează:
    - Modelul compilat
    """
    model = tf.keras.Sequential([
        # Prima convoluție cu regularizare L2
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                              kernel_regularizer=l2(0.001),
                              input_shape=(128, 128, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # A doua convoluție cu regularizare L2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),  # Adaugă dropout după al doilea strat
        
        # A treia convoluție cu regularizare L2
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # A patra convoluție pentru capacitate mai mare
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),  # Adaugă dropout după al patrulea strat
        
        # Dense layers cu regularizare
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),  # Dropout mai agresiv pentru stratul dens
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        # Modificare: Inițializează ultimul strat cu bias către Real (clasa 1)
        tf.keras.layers.Dense(1, activation='sigmoid', 
                            bias_initializer=tf.keras.initializers.Constant(1.0))
    ])
    
    # Compilare model cu learning rate personalizat
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("Model creat cu succes - versiune îmbunătățită cu regularizare și bias inițial")
    model.summary()
    
    return model

def pregateste_date(director_dataset, dimensiune_imagine=128, batch_size=32, validare_split=0.2):
    """
    Pregătește datele pentru antrenare cu augmentare și balansare a claselor
    
    Parametri:
    - director_dataset: Calea către directorul cu date
    - dimensiune_imagine: Dimensiunea la care să fie redimensionate imaginile
    - batch_size: Mărimea batch-ului
    - validare_split: Procentul de date folosite pentru validare
    
    Returnează:
    - Generatorii de date pentru antrenare și validare
    """
    # Verifică existența directorului
    if not os.path.exists(director_dataset):
        raise FileNotFoundError(f"Directorul {director_dataset} nu există!")
    
    # Verifică existența subdirectoarelor pentru categorii
    dir_fake = os.path.join(director_dataset, "fake")
    dir_real = os.path.join(director_dataset, "real")
    
    if not (os.path.exists(dir_fake) and os.path.exists(dir_real)):
        raise FileNotFoundError(f"Subdirectoarele 'fake' și 'real' trebuie să existe în {director_dataset}")
    
    # Numărăm fișierele pentru a calcula class weights
    nr_fake = len(os.listdir(dir_fake))
    nr_real = len(os.listdir(dir_real))
    total = nr_fake + nr_real
    
    print(f"Distribuția datelor: {nr_fake} imagini fake, {nr_real} imagini reale")
    
    # Calculăm ponderi pentru clase pentru a echilibra setul de date
    class_weight = {
        0: 1.0,  # Clasa 'fake'
        1: 3.0   # Clasa 'real' - pondere mai mare pentru a favoriza predicția Real
    }
    
    if nr_fake > 0 and nr_real > 0:
        # Ajustăm ponderile pentru a echilibra clasele, favorizând puternic Real
        class_weight[0] = 1.0
        class_weight[1] = 5.0  # Pondere foarte mare pentru clasa Real
    
    print(f"Ponderi clase pentru echilibrare: fake={class_weight[0]:.2f}, real={class_weight[1]:.2f}")
    
    # Configurare augmentare pentru date de antrenare (mai agresivă)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,             # Rotație mai amplă
        width_shift_range=0.3,         # Translație mai amplă pe orizontală
        height_shift_range=0.3,        # Translație mai amplă pe verticală
        shear_range=0.3,               # Deformare mai amplă
        zoom_range=0.3,                # Zoom mai amplu
        horizontal_flip=True,
        vertical_flip=True,            # Adăugat flip vertical
        brightness_range=[0.7, 1.3],   # Variații de luminozitate
        fill_mode='nearest',
        validation_split=validare_split
    )
    
    # Datagen pentru validare - doar rescale
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validare_split
    )
    
    # Generator pentru date de antrenare
    train_generator = train_datagen.flow_from_directory(
        director_dataset,
        target_size=(dimensiune_imagine, dimensiune_imagine),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Generator pentru date de validare
    validation_generator = validation_datagen.flow_from_directory(
        director_dataset,
        target_size=(dimensiune_imagine, dimensiune_imagine),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )
    
    print(f"Date pregătite: {train_generator.samples} imagini pentru antrenare, "
          f"{validation_generator.samples} imagini pentru validare")
    
    return train_generator, validation_generator, class_weight

def antreneaza_model(model, train_generator, validation_generator, epoci=30, model_path="model/model_trained.h5", class_weight=None):
    """
    Antrenează modelul cu mai multe callback-uri și strategii de optimizare
    
    Parametri:
    - model: Modelul de antrenat
    - train_generator: Generatorul de date pentru antrenare
    - validation_generator: Generatorul de date pentru validare
    - epoci: Numărul de epoci
    - model_path: Calea unde să fie salvat modelul antrenat
    - class_weight: Ponderi pentru clase pentru a echilibra setul de date
    
    Returnează:
    - Istoricul antrenării
    - Calea către modelul salvat
    """
    # Creăm directorul pentru model dacă nu există
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Configurăm callbacks pentru antrenare
    callbacks = [
        # Oprire timpurie dacă modelul nu se îmbunătățește
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,  # Mai multă răbdare
            restore_best_weights=True,
            verbose=1
        ),
        # Salvare model când se îmbunătățește performanța
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reducerea ratei de învățare când performanța stagnează
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # Reducere mai agresivă
            patience=5,  # Mai multă răbdare
            min_lr=0.000001,  # Limită inferioară mai mică
            verbose=1
        ),
        # TensorBoard pentru vizualizarea antrenării
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    print(f"Începe antrenarea pentru {epoci} epoci...")
    if class_weight:
        print(f"Folosesc ponderare pentru clase: {class_weight}")
    
    # Antrenăm modelul
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epoci,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight  # Adăugăm ponderarea claselor
    )
    
    print(f"Antrenare finalizată. Modelul a fost salvat la: {model_path}")
    
    return history, model_path

def afiseaza_rezultate_antrenare(history):
    """
    Afișează graficele pentru acuratețe și pierdere
    
    Parametri:
    - history: Istoricul antrenării
    """
    # Verifică dacă există istoricul
    if history is None or not hasattr(history, 'history'):
        print("Nu există istoric pentru afișare")
        return
    
    # Extrage valorile pentru acuratețe și pierdere
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history.get('precision', [])
    val_precision = history.history.get('val_precision', [])
    recall = history.history.get('recall', [])
    val_recall = history.history.get('val_recall', [])
    
    epoci = range(1, len(acc) + 1)
    
    # Creăm figura cu patru subploturi pentru mai multe metrici
    plt.figure(figsize=(16, 10))
    
    # Subplot pentru acuratețe
    plt.subplot(2, 2, 1)
    plt.plot(epoci, acc, 'b-', label='Acuratețe antrenare')
    plt.plot(epoci, val_acc, 'r-', label='Acuratețe validare')
    plt.title('Acuratețe în timpul antrenării')
    plt.legend()
    
    # Subplot pentru pierdere
    plt.subplot(2, 2, 2)
    plt.plot(epoci, loss, 'b-', label='Pierdere antrenare')
    plt.plot(epoci, val_loss, 'r-', label='Pierdere validare')
    plt.title('Pierdere în timpul antrenării')
    plt.legend()
    
    # Subplot pentru precizie, dacă există
    if precision and val_precision:
        plt.subplot(2, 2, 3)
        plt.plot(epoci, precision, 'b-', label='Precizie antrenare')
        plt.plot(epoci, val_precision, 'r-', label='Precizie validare')
        plt.title('Precizie în timpul antrenării')
        plt.legend()
    
    # Subplot pentru recall, dacă există
    if recall and val_recall:
        plt.subplot(2, 2, 4)
        plt.plot(epoci, recall, 'b-', label='Recall antrenare')
        plt.plot(epoci, val_recall, 'r-', label='Recall validare')
        plt.title('Recall în timpul antrenării')
        plt.legend()
    
    # Salvează graficul
    plt.tight_layout()
    plt.savefig('model/rezultate_antrenare.png')
    print("Graficele rezultatelor au fost salvate în 'model/rezultate_antrenare.png'")
    
    # Afișează ultimele valori
    print(f"Rezultate finale:")
    print(f"  Acuratețe antrenare: {acc[-1]:.4f}")
    print(f"  Acuratețe validare: {val_acc[-1]:.4f}")
    print(f"  Pierdere antrenare: {loss[-1]:.4f}")
    print(f"  Pierdere validare: {val_loss[-1]:.4f}")
    
    if precision and val_precision:
        print(f"  Precizie antrenare: {precision[-1]:.4f}")
        print(f"  Precizie validare: {val_precision[-1]:.4f}")
    
    if recall and val_recall:
        print(f"  Recall antrenare: {recall[-1]:.4f}")
        print(f"  Recall validare: {val_recall[-1]:.4f}")

def creeaza_model_tflite(model, model_tflite, prag=0.5):
    """
    Convertește modelul în format TFLite
    
    Parametri:
    - model: Modelul Keras
    - model_tflite: Calea unde va fi salvat modelul TFLite
    - prag: Pragul pentru clasificare (implicit 0.5 pentru o clasificare mai echilibrată)
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
    parser = argparse.ArgumentParser(description="Crează și antrenează model pentru detectarea etichetelor fake/real")
    parser.add_argument("--dataset", type=str, default="dataset", help="Directorul cu setul de date (implicit: 'dataset')")
    parser.add_argument("--epoci", type=int, default=50, help="Numărul de epoci pentru antrenare (implicit: 50)")
    parser.add_argument("--batch_size", type=int, default=16, help="Mărimea batch-ului (implicit: 16)")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Rata de învățare (implicit: 0.0001)")
    parser.add_argument("--prag", type=float, default=0.3, help="Pragul pentru clasificare (implicit: 0.3)")
    parser.add_argument("--fara_antrenare", action="store_true", help="Doar creează modelul fără antrenare")
    args = parser.parse_args()
    
    print(f"Creez un model nou îmbunătățit compatibil cu TensorFlow {tf.__version__}")
    
    # Creăm modelul cu rata de învățare specificată
    model = creeaza_model_nou(learning_rate=args.learning_rate)
    
    if not args.fara_antrenare:
        try:
            # Pregătim datele
            train_generator, validation_generator, class_weight = pregateste_date(
                args.dataset, 
                batch_size=args.batch_size
            )
            
            # Antrenăm modelul
            history, model_path = antreneaza_model(
                model, 
                train_generator, 
                validation_generator, 
                epoci=args.epoci,
                class_weight=class_weight
            )
            
            # Afișăm rezultatele antrenării
            afiseaza_rezultate_antrenare(history)
            
            # Reîncărcăm modelul antrenat (cel mai bun salvat)
            if os.path.exists(model_path):
                print(f"Încarc modelul antrenat: {model_path}")
                model = tf.keras.models.load_model(model_path)
            
        except FileNotFoundError as e:
            print(f"EROARE: {e}")
            print("Continuă cu crearea modelului neantrenat...")
    else:
        print("Opțiunea --fara_antrenare a fost specificată. Se omite antrenarea.")
    
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
    if not args.fara_antrenare:
        print("Modelul a fost antrenat și salvat în formatele necesare.")
    else:
        print("Un model neantrenat a fost creat și salvat în formatele necesare.")
    print("Utilizare model în Android: Consultați fișierul 'ghid_android.md'") 