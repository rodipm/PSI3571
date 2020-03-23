import numpy as np
import pandas as pd

x1_list = []
x2_list = []
x3_list = []
x4_list = []
x5_list = []
x6_list = []
y_list = []

for _ in range(50):
    # loc = media, scale = variancia
    x1 = np.random.normal(loc=10.5, scale=12)
    x2 = np.random.normal(loc=-23.123, scale=21.3)
    x3 = np.random.normal(loc=41.77, scale=7.56)
    x4 = np.random.normal(loc=-36.54, scale=44.44)
    x5 = np.random.normal(loc=64.123, scale=31.51)
    x6 = np.random.normal(loc=0, scale=1.0)

    # baseada no numero usp 9348877

    y = 9*x1 + 3*x2 + 4*x3 + 8*x4 + 8*x5 + 7*x6

    # print(x1, x2, x3, x4, x5, x6, f)
    x1_list.append(x1)
    x2_list.append(x2)
    x3_list.append(x3)
    x4_list.append(x4)
    x5_list.append(x5)
    x6_list.append(x6)
    y_list.append(y)

df = pd.DataFrame({'x1': x1_list,
                    'x2': x2_list,
                    'x3': x3_list,
                    'x4': x4_list,
                    'x5': x5_list,
                    'x6': x6_list,
                    'y': y_list})
df.to_csv("gerados2.csv", index=False)