def erro_absoluto_medio(y_real, y_previsto):
	soma_erro = 0.0
	for i in range(len(y_real)):
		soma_erro += abs(y_previsto[i] - y_real[i])
	return sum_error / float(len(y_real))

y_real = [0.1, 0.2, 0.3, 0.4, 0.5]
y_previsto = [0.11, 0.19, 0.29, 0.41, 0.5]
EAM = erro_absoluto_medio(y_real, y_previsto)
print(EAM)