import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Dimensiunea la care redimensionăm imaginile
IMG_SIZE = 128

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        # Verifică dacă fișierul este o imagine validă
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalizare
                images.append(img)
                labels.append(label)
    return images, labels

# Încarcă datele
real_images, real_labels = load_images("dataset/Poze_Reale", 0)
fake_images, fake_labels = load_images("dataset/Poze_Fake", 1)

# Combină și împarte datele
X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crearea modelului
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Clasificare binară
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Antrenare
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Salvare model
model.save("model/model_trained.h5")
print("Modelul a fost antrenat și salvat!")
