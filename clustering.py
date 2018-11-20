import numpy as np
import scipy.linalg as linalg
import sklearn.cluster as cluster
from graph_display import *
from core import *

#Pour le graphe de similarite, on va prendre l'inverse de la matrice de distance (pour essayer) ainsi
#que la matrice d'adjacence 

class ClusteringManager():
	def __init__(self, G, k):
		self.G = G
		self.k = k

	def calculer_similarite_fw(self, G):
		distances = floyd_warshall(G) + 1 #on ajoute un car pour un meme sommet, distances[i, i] = 0
		#et l'inverse buguera
		return (1/distances)

	def calculer_laplacien(self, S):
		diagonale = np.sum(S, axis=0)
		return (np.diag(diagonale) - S)

	def calculer_k_eig(self, L, k):
		w, vr = linalg.eig(L) #note : vr contient les vecteurs propres en tant que colonnes
		w     = np.real(w)
		indexes = np.argsort(w)[:k] #on prend les k premieres vp
		w, vr = w[indexes], np.array([l[indexes] for l in vr])
		print("shape : ", vr.shape)
		return w, vr #on extrait les k premieres lignes de vr

	def clusters(self, vr, k):
		kmeans = cluster.KMeans(n_clusters=k).fit(vr)
		return kmeans.labels_

	def calculer_clustering_G(self):
		G, k = self.G, self.k
		#S = self.calculer_similarite_fw(G)
		S = G
		L = self.calculer_laplacien(S)
		w, vr = self.calculer_k_eig(L, k)
		self.labels = self.clusters(vr, k)
		ga = GestionnaireAffichage(G)
		M = ga.calculer_affichage_optimise()
		ga.afficher_points(M, debug=False, labels=self.labels)