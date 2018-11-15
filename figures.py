import core
np = core.np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def estimation_probas_degres(N, M, show=False):
	degres_sommets = np.zeros((N, N))
	#matrice de taille N x N : l'element (i, j) est le nombre de graphes (entre 0 et M) dans lequel
	#le sommet i a eu j aretes
	for _ in range(M):
		G = core.construire_G(N)
		degres = np.sum(G, axis=0).astype(int)
		for i in range(N):
			d = degres[i]
			degres_sommets[i, d] += 1.0
	for i in range(N):
		degres_sommets[i] = degres_sommets[i]/np.linalg.norm(degres_sommets[i])
	probas_degres = np.average(degres_sommets, axis=0)
	if show:
		plt.plot(np.linspace(1, N, N), probas_degres)
		plt.show()
	return probas_degres

def calcul_nombre_degres(N, M, show=False):
	#on fait avec un seul graphe
	nb_s_degres= [0]*N #nombre de sommets qui ont le degre s 
	for _ in range(M):
		G = core.construire_G(N)
		degres = np.sum(G, axis=0).astype(int) #stocke les degres de chaque sommet 
		for d in degres: #pour chaque sommet, on regarde son degre d 
			nb_s_degres[d] += 1
	nb_s_degres = np.array(nb_s_degres)
	nb_s_degres = list(nb_s_degres/np.sum(nb_s_degres))
	tmp = [(i, nb_s_degres[i]) for i in range(len(nb_s_degres)) if nb_s_degres[i] != 0]
	X = np.array([t[0] for t in tmp])
	Y = np.array([t[1] for t in tmp])
	Y = [Y[i] for i in range(len(X))] #if X[i] < 50"""
	X = [x for x in X] #if x < 50"""
	if show:	
		plt.plot(np.log(X), np.log(Y))
		plt.show()
	return np.log(X), np.log(Y)

def regression(log_X, log_Y):
	regr = LinearRegression()
	regr.fit(log_Y, log_X)
	return regr.coef_

if __name__ == '__main__':
	N, M = 5000, 500
	probas_degres = estimation_probas_degres(N, M, show=False)
	log_progas = np.log(probas_degres)
	log_linspace = np.log(np.linspace(1, N, N))
	plt.plot(log_linspace, log_progas)
	plt.show()
