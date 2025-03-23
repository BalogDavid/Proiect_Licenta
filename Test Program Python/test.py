import tensorflow as tf
import numpy as np

# Încarcă modelul antrenat
model = tf.keras.models.load_model("model/model_trained.h5")

# Evaluează pe setul de testare
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acuratețea modelului: {accuracy * 100:.2f}%")
