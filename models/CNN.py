#####librerias
import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

##### para afinamiento #########
import keras_tuner as kt

# Para cargar salidas
import joblib
import os


os.chdir(r'C:\Users\amqj1\Downloads\salud')

# Separamiento entre entrenamiento y testeo
from sklearn.model_selection import train_test_split

######## cargar datos #####
# Cambiar rutas segun la maquina
x = joblib.load('salidas\\x.joblib')
y = joblib.load('salidas\\y.joblib')

# Separar datos entre entrenamiento y testeo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


x_train.shape
x_train[1]
x_train.max()
x_train.min()

y_train.shape


plt.imshow(x_train[1],cmap='gray')
y_train[1]


# Normalizar datos

x_trains = x_train / 255
x_tests = x_test / 255

x_train[1]
x_train.max()

f=x_train.shape[1]
c=x_train.shape[2]
fxc= f*c

#########################################################
######### red convolucional #############################
#########################################################


from tensorflow import keras

fa = 'tanh'  # Función de activación

cnn1 = keras.models.Sequential([
    # Primera capa convolucional
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=fa, input_shape=(f, c, 1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=fa),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Segunda sección de capas convolucionales
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=fa),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=fa),
    keras.layers.MaxPooling2D(pool_size=(2,2)),

    # Tercera sección de capas convolucionales
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=fa),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Aplanado y capas densas con Dropout
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),  # Dropout para evitar el sobreajuste
    keras.layers.Dense(256, activation=fa, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=fa, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    
    # Capa de salida
    keras.layers.Dense(1, activation="sigmoid")  # Sigmoid para clasificación binaria
])


cnn1.summary()


lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
optim=keras.optimizers.Adam(lr) ### se configar el optimizador
cnn1.compile(optimizer=optim, loss="mean_squared_error", metrics=['mse'])
# cnn1.fit(x_trains, y_train, epochs=5,validation_data=(x_tests, y_test), batch_size=30)

# Entrenar el modelo y guardar el historial
history = cnn1.fit(x_trains, y_train, epochs=5, validation_data=(x_tests, y_test), batch_size=30)

# GUardar history en archivo joblib

joblib.dump(history.history, 'salidas\\history.joblib')

# Graficar el mse a traves de los epochs
plt.plot(history.history['mse'])
# plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()


# Ver el desempeño de cnn1
test_loss, test_acc = cnn1.evaluate(x_tests, y_test)

print("Test mse: ", round(test_loss, 2))
print("Test mae: ", round(test_acc, 2))



############################################################
#########  Afinamiento de hiperparámetros ##################
###########################################################

hp=kt.HyperParameters()

num_conv_layers=3
def build_model(hp):
    
    ####### definición de hiperparámetros de grilla 
    
    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3)
    conv_filters = [hp.Int(f'conv_{i+1}_filter', min_value=1, max_value=32, step=16) for i in range(num_conv_layers)]
    conv_kernels = [hp.Choice(f'conv_{i+1}_kernel', values=[3, 1]) for i in range( num_conv_layers)]
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    dense_units = hp.Int('dense_units', min_value=8, max_value=64, step=32) 
    
    ####### creación de modelo sequential vacío y capa de entrada

    model = keras.models.Sequential()### se crea modelo sin ninguna capa
    model.add(keras.layers.InputLayer(input_shape=(f, c, 1))) ### se crea capa de entrada
    
    ##### agregar capas convolucionales de acuerdo a hiperparáemtro de capas
    
    for i in range( num_conv_layers):
        model.add(keras.layers.Conv2D(filters=conv_filters[i], kernel_size=(conv_kernels[i], conv_kernels[i]), activation=activation))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    
    ### agregar capas densas siempre estándar al final de la red 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=dense_units, activation=activation))
    model.add(keras.layers.Dense(1, activation='relu'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_absolute_error', metrics=['mse'])
    
    return model

tuner = kt.RandomSearch(
    build_model,
    hyperparameters=hp,
    objective='val_m',
    max_trials=4,
    directory='my_dir',
    overwrite=True,
    project_name='cnn_tuning'
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=600)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model.summary()
best_hps.values