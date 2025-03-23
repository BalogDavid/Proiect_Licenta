import tensorflow as tf
import numpy as np
import cv2
import sys
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setarea dimensiunii imaginii
IMG_SIZE = 128

def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Adaugă dimensiunea batch-ului

    prediction = model.predict(img)[0][0]
    return "Fake" if prediction > 0.5 else "Real", prediction

def evaluate_model(model, test_directory):
    # Creează generatorul pentru setul de testare
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='binary'  # 2 clase: Real și Fake
    )

    # Predicții pentru întregul set de testare
    predictions = model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype("int32")

    # Etichetele reale
    true_classes = test_generator.classes

    # Calcularea matricei de confuzie
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Matricea de Confuzie:")
    print(cm)

    # Afișarea matricei de confuzie fără seaborn
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matricea de Confuzie')
    plt.colorbar()
    tick_marks = np.arange(len(test_generator.class_indices))
    plt.xticks(tick_marks, test_generator.class_indices.keys())
    plt.yticks(tick_marks, test_generator.class_indices.keys())
    plt.ylabel('Realități')
    plt.xlabel('Predicții')
    plt.show()

    # Raport de clasificare (precizie, recall, F1-score)
    print("Raportul de clasificare:")
    print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))

    # Calcularea acurateței
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Acuratețea pe setul de testare: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Dacă dai un argument pentru imaginea de test, face predicția pentru acea imagine
    if len(sys.argv) == 2:
        image_path = sys.argv[1]  # Primește imaginea ca argument
        model = tf.keras.models.load_model("model/model_trained.h5")
        result, prediction_score = predict_image(image_path, model)
        print(f"Rezultatul clasificării pentru {image_path}: {result}")
        print(f"Probabilitatea predicției: {prediction_score * 100:.2f}%")

    # Dacă dai un argument pentru a evalua întregul set de testare
    elif len(sys.argv) == 3 and sys.argv[1] == "evaluate":
        test_directory = sys.argv[2]  # Calea către folderul cu imagini de testare
        model = tf.keras.models.load_model("model/model_trained.h5")
        evaluate_model(model, test_directory)
    else:
        print("Utilizare:")
        print("  python predict.py <cale_imagine>  - pentru a face o predicție pe o imagine individuală.")
        print("  python predict.py evaluate <cale_folder_testare> - pentru a evalua modelul pe setul de testare.")
