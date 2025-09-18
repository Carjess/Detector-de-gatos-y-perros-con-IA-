import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Cargar modelo
model = tf.keras.models.load_model('modelo_3clases.h5')
class_names = ['gato', 'humano', 'perro']  # Asegúrate que coincida con tu entrenamiento

def predecir_imagen(ruta_imagen):
    try:
        # Cargar imagen
        img = image.load_img(ruta_imagen, target_size=(160, 160))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicción
        predictions = model.predict(img_array)[0]
        max_idx = np.argmax(predictions)
        max_confidence = predictions[max_idx] * 100
        
        # Lógica mejorada de clasificación
        if max_confidence > 85:  # Alto nivel de confianza
            return f"✅ Es claramente un {class_names[max_idx]} ({max_confidence:.1f}% de confianza)"
        elif max_confidence > 60:
            if class_names[max_idx] == 'otros':
                return f" Es una persona ({max_confidence:.1f}%)"
            else:
                return f" Es un {class_names[max_idx]} ({max_confidence:.1f}%)"
        else:
            # Mostrar las 3 probabilidades
            results = []
            for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
                results.append(f"{class_name}: {prob*100:.1f}%")
            return "🔍 Resultados:\n" + "\n".join(results)
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Ejemplo de uso
print(predecir_imagen("pruebas/jesus1.jpg"))