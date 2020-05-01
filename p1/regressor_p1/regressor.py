import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras import backend as K

# Junta as tabelas de estudantes
d1 = pd.read_csv("student-mat.csv", sep=";")
d2 = pd.read_csv("student-por.csv", sep=";")
dataset = pd.concat([d1, d2])

# Efetua OneHotEncoding
cols_to_encode = [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22] # Colunas com valores binários ou categoricos
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), cols_to_encode)],remainder='passthrough')
dataset=pd.DataFrame(ct.fit_transform(dataset))

# Separa entre train e test
train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Pega os X (entrada) e Y (saida)
X_train = train_dataset.iloc[:, :-1].values
Y_train = train_dataset.iloc[:, -1].values
X_test = test_dataset.iloc[:, :-1].values
Y_test = test_dataset.iloc[:, -1].values

# Cria o modelo de regressao
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="mae", optimizer=optimizer, metrics=["mae", "mse"])


#Efetua o treinamento da rede neural
model.fit(X_train, Y_train, epochs=600, validation_split=0.2, verbose=0)

# Verifica acuracia com dataset de testes
loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)

print("Loss: ", loss, "Mae: ", mae, "Mse: ", mse)

# # Testes MAE e MAEN

test_batch = X_test[:]
test_ys = Y_test[:]

predicted_ys = model.predict(test_batch)

for i, test_y in enumerate(test_ys):
    print("Valor original: ", test_y, "Valor do regressor: ", predicted_ys[i][0])

def erro_absoluto_medio(y_real, y_previsto):
        soma_erro = 0.0
        for i in range(len(y_real)):
                soma_erro += abs(y_previsto[i] - y_real[i])
        return soma_erro / float(len(y_real))

def erro_absoluto_medio_negativo(y_real, y_previsto):
    soma_erro = 0.0
    for i in range(len(y_real)):
        if (y_previsto[i] - y_real[i] < 0):
            soma_erro += abs(y_previsto[i] - y_real[i])
    return soma_erro/float(len(y_real))

print("EAM:", erro_absoluto_medio(test_ys, predicted_ys))
print("EAMN:", erro_absoluto_medio_negativo(test_ys, predicted_ys))