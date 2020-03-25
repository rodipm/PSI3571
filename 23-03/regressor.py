import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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