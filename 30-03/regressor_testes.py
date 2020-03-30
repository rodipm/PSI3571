import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

raw_dataset = pd.read_csv("gerados.csv")

dataset = raw_dataset.copy()

# Remove o campo de variância aleatória (x6)
dataset.pop('x6')

# Separa entre train e test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separa os valores objetivos 
train_labels = train_dataset.pop('y')
test_labels = test_dataset.pop('y')

# Cria o modelo de regressao
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# optimizer = tf.keras.optimizers.RMSprop(0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss="mae", optimizer=optimizer, metrics=["mae", "mse"])

#Efetua o treinamento da rede neural

model.fit(train_dataset, train_labels, epochs=600, validation_split=0.2, verbose=0)


# Verifica acurácia com dataset de testes
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

print("Loss: ", loss, "Mae: ", mae, "Mse: ", mse)

test_batch = test_dataset[:10]
test_ys = test_labels[:10]
test_results = model.predict(test_batch)

for i, test_y in enumerate(test_ys):
    print("Valor original: ", test_y, "Valor do regressor: ", test_results[i][0])

# Testes MAE e MAEN
testes = 25
y_real = []
y_previsto = []
for _ in range(testes):
    # loc = media, scale = variancia
    x1 = np.random.normal(loc=10.5, scale=12)
    x2 = np.random.normal(loc=-23.123, scale=21.3)
    x3 = np.random.normal(loc=41.77, scale=7.56)
    x4 = np.random.normal(loc=-36.54, scale=44.44)
    x5 = np.random.normal(loc=64.123, scale=31.51)
    x6 = np.random.normal(loc=0, scale=1.0)

    test_input = pd.DataFrame([[x1, x2, x3, x4, x5]])

    # baseada no numero usp 9348877
    y = 9*x1 + 3*x2 + 4*x3 + 8*x4 + 8*x5 + 7*x6

    prediction = (model.predict(test_input)[0][0])

    y_real.append(y)
    y_previsto.append(prediction)

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

print("EAM:", erro_absoluto_medio(y_real, y_previsto))
print("EAMN:", erro_absoluto_medio_negativo(y_real, y_previsto))