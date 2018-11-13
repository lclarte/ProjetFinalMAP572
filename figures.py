import core
np = core.np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calcul_nombre_degres(N, M, show=False):
	#on fait avec un seul graphe
	nb_s_degres= [0]*N #nombre de sommets qui ont le degre s 
	for _ in range(M):
		G = core.construire_G(N)
		degres = np.sum(G, axis=0).astype(int) #stocke les degres de chaque sommet 
		for d in degres: #pour chaque sommet, on regarde son degre d 
			nb_s_degres[d] += 1
	tmp = [(i, nb_s_degres[i]) for i in range(len(nb_s_degres)) if nb_s_degres[i] != 0]
	X = np.array([t[0] for t in tmp])
	Y = np.array([t[1] for t in tmp])
	Y = [Y[i] for i in range(len(X)) if X[i] > 10 and X[i] < 50] #if X[i] < 50"""
	X = [x for x in X if x > 10 and x < 50] #if x < 50"""
	if show:	
		plt.plot(np.log(X), np.log(Y))
		plt.show()
	return np.log(X), np.log(Y)

def regression(log_X, log_Y):
	regr = LinearRegression()
	regr.fit(log_Y, log_X)
	return regr.coef_

lX, lY = calcul_nombre_degres(2000, 10, True)
lX = [[x] for x in lX]
lY = [[y] for y in lY]
print(lX)
coef = regression(lX, lY)
print(coef) #je trouve quelque chose entre 0.4 et 0.5
#attention : la loi est pas vraie pour les k trop petits
