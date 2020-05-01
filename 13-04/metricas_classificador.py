import numpy as np

cm = np.random.rand(2, 2)

def acc(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()

def sensi(linha, confusion_matrix):
    row = confusion_matrix[linha, :]
    return confusion_matrix[linha, linha] / row.sum()

print(cm)
print(acc(cm))
print(sensi(0, cm))
