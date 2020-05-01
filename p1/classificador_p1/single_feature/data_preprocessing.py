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
    # obtém os dados doa arquivo sendo processado
    fold_no=str(data.iloc[i]["fold"])
    file=data.iloc[i]["slice_file_name"]
    label=data.iloc[i]["classID"]
    filename=path+fold_no+"/"+file
    print(filename)

    # Processamento do arquivo de audio utilizando o feature mfcc
    y,sr=librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    print(mfccs.shape,mfccs.max(),mfccs.min())

    # A pasta 10 será utilizada para o grupo de testes
    if(fold_no!='10'):
      x_train.append(mfccs)
      y_train.append(label)
    else:
      x_test.append(mfccs)
      y_test.append(label)

# Gera numpy arrays a partir dos dados para serem salvos
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

# Cria os arquivos tratados
np.savetxt("train_data.csv", x_train, delimiter=",")
np.savetxt("test_data.csv",x_test,delimiter=",")
np.savetxt("train_labels.csv",y_train,delimiter=",")
np.savetxt("test_labels.csv",y_test,delimiter=",")

# from google.colab import files

# files.download("train_data.csv")
# files.download("test_data.csv")
# files.download("train_labels.csv")
# files.download("test_labels.csv")