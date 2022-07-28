# Para lectura y manipulación de datos
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# Modelado
# necesitarás keras https://keras.io/
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# read the database
# usaremos el iris dataset
# https://en.wikipedia.org/wiki/Iris_flower_data_set
df = pd.read_csv('./iris.csv')

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)
# print the dataset para ver los datos
print(df)


# Preparamos los datos para el MODELADO
# 1. split the data
# 2. convertimos a numpy arrays para usar keras
# 3. Codificamos las classes como enteros
# es decir, las convertimos a enteros

# split
Y = df['species']
X = df.drop(['species'], axis=1)

print(X.shape)
print(Y.shape)

# numpy arrays
X = np.array(X)

# revisamos los prrimeros 10 valores de Y 
print( Y.head(10) )


# los encoders nos permites trabajar con labels 
# pero utilizar numeros en el procesamiento
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#aquí veremos 0 1 2 en lugar de las 3 especies
print(encoded_Y)
# aqui veremos vectores de la forma [0. 0. 1.]
# que se utilizan para la Neural Net la posision del 1 determina la clase de la flor
print(dummy_y)



# Build, Compile and Fit model



# build a model

model = Sequential()
#  podemos revisar la full documentation aqui https://keras.io/guides/sequential_model/

model.add(Dense(16, input_shape=(X.shape[1],), activation='relu')) # input shape is (features,)
model.add(Dense(3, activation='softmax'))
model.summary()

# compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# early stopping callback
# este callback detendrá el entrenamiento cuando NO hay mejora en la validation loss (val_loss) por 10 épocas consecutivas.
# aquí podemos entender EarlyStopping https://keras.io/api/callbacks/early_stopping/

# En la pagina podemos leer: Stop training when a monitored metric has stopped improving.

# guardamos la callback en una variable llamada "es" 
es = keras.callbacks.EarlyStopping(monitor='val_loss',# metrica que vamos a monitorear
                                   mode='min',
                                   patience=10, # numero de epocas que esperamos
                                   restore_best_weights=True) 

# update our model fit call
history = model.fit(X,
                    dummy_y,
                    callbacks=[es],
                    epochs=3000, # Numero máximo de épocas
                    batch_size=10,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)



# Esta parte es para visualizar el entrenamiento
history_dict = history.history

# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
plt.plot(epochs, acc, 'red', label='Training accuracy')
plt.plot(epochs, val_acc, 'blue', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#---------------------------------------------------------


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

preds = model.predict(X) # vemos como ha funcionado el modelo!
print(preds)

print("--------------------------")
# la clase es el valor mas alto
print( np.argmax(preds,axis=1) )

# Almost a perfect prediction
# actual is left, predicted is top
# names can be found by inspecting Y
matrix = confusion_matrix(dummy_y.argmax(axis=1), preds.argmax(axis=1))
print(matrix)
#imprimimos la matriz de confusion


# classification report!
print(classification_report(dummy_y.argmax(axis=1), preds.argmax(axis=1)))














