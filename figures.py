import core
np = core.np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calcul_moyenne_degres(N, M, show=False):
	degres = np.zeros((M, N))
	for _ in range(M):
		G = core.construire_G(N)
		degres[_] = np.sum(G, axis=0)
	degres_average = np.average(degres, axis=0)
	if show:
		plt.plot(np.linspace(0, N, N), degres_average)
		plt.show()
	return degres_average

def log_degres_log_k(N, M, N_tronque=0, show=False):
	log_degres = np.zeros((M, N))
	for _ in range(M):
		G = core.construire_G(N)
		log_degres[_] = np.log(np.sum(G, axis=0))
	log_degres_avg = np.average(log_degres, axis=0)
	if show:
		plt.clf()
		X = np.linspace(N_tronque, N, N - N_tronque)
		log_k = np.log(X)
		plt.plot(log_k, log_degres_avg[N_tronque:N])
		plt.show()
	return log_degres_avg

log_degres_log_k(500, 500, 0, True)