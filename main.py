import numpy as np

#structure de donnees : matrice d'adjacence 
def recurrence_G(matrice):
	n = len(matrice)
	matrice2 = np.zeros((n+1, n+1))
	matrice2[:n, :n] = matrice
	probas = np.sum(matrice, axis=0)/(2*n-1)
	v = np.random.choice(n, p=probas)
	matrice2[v, n] = matrice2[n, v] = 1
	return matrice2

def construire_G(n):
	matrice = np.array([[1]])
	for _ in range(n-1):
		matrice = recurrence_G(matrice)
	return matrice

print(construire_G(10))