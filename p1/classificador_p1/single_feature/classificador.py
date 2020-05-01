from numpy import genfromtxt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense,Dropout,Activation
from keras import metrics

# Carrega os dados pre processados
x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')

# Transformação dos dados para vetores categoricos de 10 classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Criação do modelo
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=40))
model.add(Dropout(0.4))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=10, activation='softmax'))

# Compilação e treinamento
model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),batch_size=50)

train_loss_score = model.evaluate(x_train, y_train)
test_loss_score = model.evaluate(x_test, y_test)
print(f"TREINO: Acuracia: {train_loss_score[1]} - Loss {train_loss_score[0]}")
print(f"TESTE: Acuracia: {test_loss_score[1]} - Loss {test_loss_score[0]}")

# Efetuar a classificação para 50 pares
testes = x_test[100:151]
testes_labels = y_test[100:151]
resultados = []

for teste in testes:
    resultados.append(np.argmax(model.predict(np.array([teste]))))

acertos = 0
for i, res in enumerate(resultados):
    print(f"Label:{np.argmax(testes_labels[i])} - Classificacao: {res}")
    if np.argmax(testes_labels[i]) == res:
        acertos += 1

print(f"Acuracia (50 pares): {acertos/len(resultados)*100}%")


# Efetuar a classificação para 1000 pares
testes = x_test[:1000]
testes_labels = y_test[:1000]
resultados = []

for teste in testes:
    resultados.append(np.argmax(model.predict(np.array([teste]))))

acertos = 0
for i, res in enumerate(resultados):
    if np.argmax(testes_labels[i]) == res:
        acertos += 1

print(f"Acuracia (1000 pares): {acertos/len(resultados)*100}%")
