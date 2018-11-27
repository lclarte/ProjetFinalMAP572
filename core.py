import numpy as np

delta = 0 #utilise dans le delta-attachement preferentiel 

#structure de donnees : matrice d'adjacence 
def recurrence_G(matrice):
	n = len(matrice)
	matrice2 = np.zeros((n+1, n+1))
	matrice2[:n, :n] = matrice
	probas = np.sum(matrice, axis=0)/(2*n-1)
	v = np.random.choice(n, p=probas)
	matrice2[v, n] = matrice2[n, v] = 1
	return matrice2

def construire_G_normal(n):
	matrice = np.array([[1]])
	for _ in range(n-1):
		matrice = recurrence_G(matrice)
	return matrice

#============= Nouvel algorithme =============

def recurrence_G_optimise(liste, degres, fn_probas=None): 
	"""fn_probas donne la probabilite du sommet par rapport a son degre et"""
	#1 : on va stocker sous forme de liste d'adjacence
	#2 : on va stocker a cote de notre liste la liste des differents degres 
	somme = sum(degres)
	assert somme == 2*len(degres) - 1
	n = len(liste)
	if fn_probas == None:#on est dans le cas normal
		probas = [float(d)/float(somme) for d in degres]
	else:
		probas = [fn_probas(d, somme) for d in degres]
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

def construire_G_optimise(n, fn_probas=None):
	liste = [[0]]
	degres = [1]
	for _ in range(n-1):
		liste, degres = recurrence_G_optimise(liste, degres, fn_probas)
	return list_to_matrix(liste)

#Stochastic Bloch-Model
def simuler_SBM(n, K, q):
	#q : matrice symetrique de taille K x K ou q[i, j] : proba de liens entre les classes i et j
	#repartition des sommets : in(n/K) sommets par classe, sauf pour la derniere qui prend le reste des 
	#divisions des autres
	matrice_adjacence = np.zeros((n, n))
	nb = int(n/K)
	for i in range(n):
		classe_i = min(int(i/nb), K-1)
		for j in range(n):
			classe_j = min(int(j/nb), K-1)
			pij, X = q[classe_i, classe_j], np.random.uniform()
			matrice_adjacence[i, j] = matrice_adjacence[j, i] = [0, 1][X <= pij]
	return matrice_adjacence

#fonction pour le delta-attachement preferentiel
def fn_probas_delta(degre, somme, delta):
	#somme = 2*k - 1 
	k = (somme + 1)/2
	return float(degre + delta)/float(somme + k*delta)

def orienter_graphe(G):
	#ici, si j > i alors j est oriente vers i donc on a juste a mettre a 0 tous les elements dans la sur diagonale
	n = len(G)
	G_or = np.copy(G)
	for i in range(n):
		for j in range(i+1, n):
			G_or[i, j] = 0
	return G_or

construire_G = construire_G_optimise
construire_G_delta = lambda n, delta: construire_G_optimise(n, fn_probas= lambda d, s: fn_probas_delta(d, s, delta))