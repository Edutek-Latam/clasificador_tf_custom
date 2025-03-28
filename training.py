import os
import tensorflow as tf
import json
from datetime import datetime

#Crear directorio para logs dde TensorBoard
log_dir = os.path.join('logs','tensorboard',datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir,exist_ok=True )

#Configuracion de parametros
train_dir = 'data/training'
validation_dir = 'data/test'
img_heigth = 150
img_width = 150
bach_size = 32
num_epochs = 50


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale= 1./255,
    rotation_range=20,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_heigth,img_width),
    batch_size=bach_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_heigth,img_width),
    batch_size=bach_size,
    class_mode='categorical',
    subset='validation'
)

#Obtener numero de clases
num_class = len(train_generator.class_indices)
print(f"Numero de clases detectadas : {num_class}")
print(f"Clases: {train_generator.class_indices}")

with open("class_mapping.json",'w') as f:
    json.dump(train_generator.class_indices,f,indent=4)

def create_model(num_class):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(img_heigth,img_width,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(num_class,activation='softmax')
    ])
    return model

#crear Modelo
num_class = len(train_generator.class_indices)
model = create_model(num_class)

#Compilar Modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,#Registra histogramas de pesos por cada epoch
    write_graph=True,# Visualizar graficos de modelo
    write_images=True,#visualizar pesos como imagenes
    update_freq='epoch'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)


#Entrnar modelo
rest = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs,
    callbacks=[tensorboard_callback,early_stopping, model_checkpoint]
)

print(f"\nLogs de TensorBoard guardado en: {log_dir}")