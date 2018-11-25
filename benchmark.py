import core, time
import numpy as np
import matplotlib.pyplot as plt

def compare_construire_G(size, nb_runs):
	times_construire_1 = np.zeros(nb_runs)
	for i in range(nb_runs):
		start = time.time()
		core.construire_G(size)
		end = time.time()
		times_construire_1[i] = end - start
	avg_1 = np.average(times_construire_1)
	times_construire_2 = np.zeros(nb_runs)
	for i in range(nb_runs):
		start = time.time()
		core.construire_G_optimise(size)
		end = time.time()
		times_construire_2[i] = end - start
	avg_2 = np.average(times_construire_2)
	return avg_1, avg_2

#cette fonction sert a calculer la variance du degre des sommets 
#pour mettre en evidence que quand delta augmente, la variance diminue
def variance_degres_delta():
	deltas = [0.5*i for i in range(100)]
	N, variances = 1000, []
	variances = []
	for d in deltas:
		print('delta = ', d)
		G = core.construire_G_delta(N, d)
		degres_sommets = np.sum(G, axis=0)
		variance = np.var(degres_sommets)
		variances.append(variance)
	X = np.linspace(1, len(variances), len(variances))
	plt.plot(X, variances)
	plt.show()

#trace le score moyen des sommets de degre d en fonction de d
def pagerank_score_degre_entrant():
	pass