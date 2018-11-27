import core
import graph_display as g_d
import numpy as np
from clustering import *
from benchmark import *
import pagerank as pr

"""
#pour tester le stochastic block model
k = 20
G = np.loadtxt("StochasticBlockModel.txt")
cm = ClusteringManager(G, k)
cm.calculer_clustering_G()
"""

"""
#pour afficher des graphes delta-preferentiel
deltas = [-0.99999, 0.0, 1000.0]
ns = [20, 20, 50]
for d, N in zip(deltas, ns):
	G = construire_G_delta(N, d)
	g = GestionnaireAffichage(G)
	M = g.calculer_affichage_optimise(verbose=False)
	g.afficher_points(M, debug=False)
"""

#pour calculer la variance des degres des sommets en fonction de delta
#variance_degres_delta()

#Pour tester le clustering sur le SBM
"""
n = 100
K = 5
q = 0.8*np.eye(K) + 0.2*np.ones(K)
G = core.simuler_SBM(n, K, q)
cm = ClusteringManager(G, K)
labels = cm.spectral_clustering(afficher=True)
print(labels)
"""

"""
#Pour tester l'effet de epsilon sur le score
epsilons = [0.01, 0.15, 0.5, 0.99]
G = core.construire_G(10)
G = core.orienter_graphe(G)
print("Graphe G : ", G)
for e in epsilons:
	vec = pr.page_rank(G, e)
	print("epsilon = ", e, "vecteur : ", np.real(np.transpose(vec)))
"""

G = core.construire_G(200)
ga = GestionnaireAffichage(G)
M = ga.calculer_affichage_optimise(method=1)
ga.afficher_points(M)