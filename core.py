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

def recurrence_G_optimise(liste, degres): 
	#1 : on va stocker sous forme de liste d'adjacence
	#2 : on va stocker a cote de notre liste la liste des differents degres 
	somme = sum(degres)
	assert somme == 2*len(degres) - 1
	n = len(liste)
	probas = [float(d)/float(somme) for d in degres]
	v = np.random.choice(n, p=probas)
	liste.append([v])
	liste[v].append(n)
	degres.append(1)
	degres[v] += 1
	return liste, degres

def list_to_matrix(liste):
	"Convertit une liste d'adjacence en matrice d'adjacence"
	n = len(liste)
	matrix = np.zeros((n, n))
	for v in range(len(liste)):
		for w in liste[v]:
			matrix[v, w] = matrix[w, v] = 1
	return matrix

def construire_G_normal(n):
	matrice = np.array([[1]])
	for _ in range(n-1):
		matrice = recurrence_G(matrice)
	return matrice

def construire_G_optimise(n):
	liste = [[0]]
	degres = [1]
	for _ in range(n-1):
		liste, degres = recurrence_G_optimise(liste, degres)
	return list_to_matrix(liste)

construire_G = construire_G_optimise