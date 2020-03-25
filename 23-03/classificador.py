import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

raw_dataset = pd.read_csv("gerados_classificacao.csv")

dataset = raw_dataset.copy()

# Remove o campo de variância aleatória (x6)
dataset.pop('x6')

# Separa entre train e test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separa os valores objetivos 
train_labels = train_dataset.pop('y_binario')
train_labels = train_labels.replace(-1, 0)
test_labels = test_dataset.pop('y_binario')
test_labels = test_labels.replace(-1, 0)

# Cria o modelo de regressao
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#Efetua o treinamento da rede neural
model.fit(train_dataset, train_labels, epochs=600, validation_split=0.2, verbose=0)


# Verifica acurácia com dataset de testes
loss, acc = model.evaluate(test_dataset, test_labels, verbose=0)

print("Loss: ", loss, "acc: ", acc)

test_batch = test_dataset[:10]
test_ys = test_labels[:10]
test_results = model.predict(test_batch)

for i, test_y in enumerate(test_ys):
    print("Valor original: ", "A" if test_y == 1.0 else "B", "Classificação: ", "A" if round(test_results[i][0]) == 1.0 else "B")


# Verificação de acurácia para novos valores
acertos = 0
testes = 1000
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
    y = np.sign(3*9*x1 - 4*3*x2 + 5*4*x3 - 6*8*x4 + 8*x5 + 7*x6)

    if y < 0:
        y = 0

    prediction = round(model.predict(test_input)[0][0])
    if y == prediction:
        acertos += 1

print("Acertos:", acertos, "/", testes)
print("Acurácia:", str((acertos/testes)*100) + "%")
    