import json
import tensorflow as tf
import os
import numpy as np

def load_class():
    try:
        with open("class_mapping.json","r") as f:
            return json.load(f)
        pass
    except:
        pass


model = tf.keras.models.load_model('best_model.h5')
clases = load_class()

#Invertir el mapeto para obtener nombres de clase por indice
clases_por_indice = {v:k for k, v in clases.items()}

def predict_image(path_img):
    img_heigth = 150
    img_width = 150

    img = tf.keras.preprocessing.image.load_img(path_img, target_size=(img_heigth,img_width))
    img_array=tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array /= 255.0

    #Predecir
    predict = model.predict(img_array)
    clase_res = np.argmax(predict[0])
    probabilidad=predict[0][clase_res]
    nombre_clase = clases_por_indice.get(clase_res,f"Clase {clase_res}")
    return nombre_clase, probabilidad

def main():
    ruta_img='perro_tesst.jpg'
    try:
        clase,probabilidad = predict_image(ruta_img)
        print(f"\n Imagen: {os.path.basename(ruta_img)}")
        print(f"Clase: {clase}")
        print(f"Probabilidad: {probabilidad}")
    except:
        pass

if __name__ == '__main__':
    main()