def erro_absoluto_medio_negativo(y_real, y_previsto):
	soma_erro_negativos = 0.0
	for i in range(len(y_real)):
        if (y_previsto[i] - y_real[i] < 0):
		    soma_erro_negativos += abs(y_previsto[i] - y_real[i])
	return soma_erro_negativos / float(len(y_real))

y_real = [0.1, 0.2, 0.3, 0.4, 0.5]
y_previsto = [0.11, 0.19, 0.29, 0.41, 0.5]
EAMN = erro_absoluto_medio_negativo(y_real, y_previsto)
print(EAMN)