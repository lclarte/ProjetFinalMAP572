import numpy as np
import scipy.linalg as linalg

def normaliser_matrice_adjacence(A):
	A_tilde = np.copy(A)
	sommes = np.sum(A_tilde, axis=1)
	n = len(A_tilde)
	for i in range(n):
		A_tilde[i] = A_tilde[i]/sommes[i]
	return A_tilde

def calculer_matrice_pagerank(A_tilde, epsilon=0.15):
	"Valeur de epsilon proposee par Page et Brin : 0.15"
	n = len(A_tilde)
	return (1-epsilon)*A_tilde + (epsilon/float(n))*np.ones((n, n))

def calculer_vecteur_pagerank(P):
	"P est la matrice P_epsilon calculee par la fonction calculer_matrice_pagerank"
	w, vl, vr = linalg.eig(P, left=True, right=True)
	w = w.astype(np.float64)
	indice = np.where(w >= 1-10**-14)
	assert len(indice) == 1
	vecteur = vl[:, indice[0]]
	vecteur.shape = (len(vecteur),1)
	return vecteur

def page_rank(G, epsilon=0.15):
	A_tilde = normaliser_matrice_adjacence(G)
	P = calculer_matrice_pagerank(A_tilde, epsilon)
	vec =  calculer_vecteur_pagerank(P)
	if vec[0, 0] < 0:
		vec *= -1
	return vec


