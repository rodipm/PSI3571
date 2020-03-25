import pandas as pd
import numpy as np

dataset = pd.read_csv("gerados.csv")
dataset.pop('y')

y_binario_list = []
for i in range(len(dataset.index)):
    x1 = dataset['x1'][i]
    x2 = dataset['x2'][i]
    x3 = dataset['x3'][i]
    x4 = dataset['x4'][i]
    x5 = dataset['x5'][i]
    x6 = dataset['x6'][i]
    y_binario_list.append(np.sign(3*9*x1 - 4*3*x2 + 5*4*x3 - 6*8*x4 + 8*x5 + 7*x6))

y_binario_df = pd.DataFrame({'y_binario': y_binario_list})

dataset = dataset.join(y_binario_df)

dataset.to_csv("gerados_classificacao.csv", index=False)