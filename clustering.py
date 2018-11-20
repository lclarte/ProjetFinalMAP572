import numpy as np
import scipy.linalg as linalg
from graph_display import *

#Pour le graphe de similarite, on va prendre l'inverse de la matrice de distance (pour essayer) ainsi
#que la matrice d'adjacence 

class ClusteringManager():
	def __init__(self, G, k):
		self.G = G
		self.k = k

	def calculer_similarite_fw(G):
		distances = floyd_warshall(G)
		return (1/distances)

	def calculer_laplacien(S):
		diagonale = np.sum(S, axis=0)
		return (np.diag(diagonale) - S)

	def calculer_k_eig(L):
		w, vr = linalg.eig(L)
		
