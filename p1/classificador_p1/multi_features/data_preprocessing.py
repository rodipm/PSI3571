import pandas as pd
import librosa
from tqdm import tqdm
import numpy as np

# Cria o dataframe do csv contendo os metadados
data=pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")

# Montando os vetores de treino e teste
x_train=[]
x_test=[]
y_train=[]
y_test=[]

# Leitura e preparação dos dados a partir da leitura dos metadados
path="UrbanSound8K/audio/fold"
for i in tqdm(range(len(data))):
    fold_no=str(data.iloc[i]["fold"]) # Obtém a pasta do áudio
    file=data.iloc[i]["slice_file_name"]  # Obtém o nome do áudio
    label=data.iloc[i]["classID"] # Obtém a classe a qual o áudio pertence
    filename=path+fold_no+"/"+file # Gera o nome completo do arquivo

    y,sr=librosa.load(filename) # Carrega o arquivo de áudio

    # Extração dos features
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))

    # Separação da última pasta para testes
    if(fold_no!='10'):
      x_train.append(features)
      y_train.append(label)
    else:
      x_test.append(features)
      y_test.append(label)

# Gera numpy arrays a partir dos dados para serem salvos
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

x_train_2d=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

# Cria os arquivos tratados
np.savetxt("train_data.csv", x_train_2d, delimiter=",")
np.savetxt("test_data.csv",x_test_2d,delimiter=",")
np.savetxt("train_labels.csv",y_train,delimiter=",")
np.savetxt("test_labels.csv",y_test,delimiter=",")