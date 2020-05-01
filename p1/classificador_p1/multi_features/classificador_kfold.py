from numpy import genfromtxt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense,Dropout,Activation
from keras import metrics

from sklearn.model_selection import KFold

# Carrega os dados pre processados
x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')


# Transformação dos dados para vetores categoricos de 10 classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

models = []
mean_acc_train = []
mean_acc_test = []

# Utilização do método K-Fold para gerar 9 modelos distintos
# Os dados de teste y_test referentes à pasta número 10 não sao utilizados para  treino
for train_index, test_index in KFold(9).split(x_train):
    # Criação do modelo
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=200))
    model.add(Dropout(0.4))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=10, activation='softmax'))

    # Compilação e treinamento
    model.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])

    # Obtenção dos grupos de teste e treinamento
    X_train, X_test = x_train[train_index], x_train[test_index]
    Y_train, Y_test = y_train[train_index], y_train[test_index]

    # Treinamento do modelo
    model.fit(X_train,Y_train,epochs=50,validation_data=(X_test,Y_test),batch_size=50, verbose=0)

    # Medidas de desempenho
    train_loss_score = model.evaluate(X_train, Y_train)
    test_loss_score = model.evaluate(X_test, Y_test)
    print(f"TREINO: Acuracia: {train_loss_score[1]} - Loss {train_loss_score[0]}")
    print(f"TESTE: Acuracia: {test_loss_score[1]} - Loss {test_loss_score[0]}")

    mean_acc_train.append(train_loss_score[1])
    mean_acc_test.append(test_loss_score[1])

    # Guarda o modelo gerado
    models.append(model)

print(np.mean(mean_acc_train))
print(np.mean(mean_acc_test))

# votacao
random_audios = np.random.randint(0, len(x_test), 100)
acertos = 0

for random_audio in random_audios:
    votacao = [0 for _ in range(10)]
    for model in models:
        vote = np.argmax(model.predict(np.array([x_test[random_audio]])))
        votacao[vote] += 1

    if np.argmax(votacao) == np.argmax(y_test[random_audio]):
        acertos += 1

    print(f"Resultado: {np.argmax(votacao)} - Label: {np.argmax(y_test[random_audio])}")
    
print(f"Acertos: {acertos/len(random_audios)}")