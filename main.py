import core
import graph_display as g_d
import numpy as np
from clustering import *
from benchmark import *
import pagerank as pr
import scipy.sparse.csgraph as csgraph

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

def tester_validite_n2():
	ns = [10, 20, 100, 200, 1000]
	Gs = [core.construire_G(n) for n in ns]
	X = np.linspace(1, 100, 100)
	for i in range(len(Gs)):
		print("n : ", ns[i])
		n = ns[i]
		g = Gs[i]
		ga = g_d.GestionnaireAffichage(g)
		ga.nb_iter = 100
		ga.delta_t = 0.0001
		M = ga.calculer_points_affichage()
		D_star = g_d.calculer_D_star(csgraph.floyd_warshall(g))
		M, grad_e_normes, energies = ga.fonction_gradient(M, D_star)
		grad_e_normes = np.array(grad_e_normes)/(float(n*n))
		plt.plot(X, grad_e_normes, label="n = " + str(n))
	legend = plt.legend(loc='upper center', shadow=False, fontsize='x-large')
	plt.show()

def tester_SBMMatrice():
	G = np.loadtxt("SBMMatrice.txt")
	ga = g_d.GestionnaireAffichage(G)
	ga.verbose = True
	ga.nb_iter = 300
	ga.delta_t = 0.0001 #TRES PETITE CONSTANTE
	ga.afficher_aretes = False
	M = ga.calculer_affichage_optimise()
	cm = ClusteringManager(G, 2)
	labels_ = cm.spectral_clustering()
	ga.afficher_points(M, label=labels_)
	return G, M

tester_validite_n2()