import core
import graph_display as g_d

#essai de differents delta 
deltas = [0, 1, 10, 1000]
n = 50
for d in deltas:
	G = core.construire_G_delta(n, d)
	g = g_d.GestionnaireAffichage(G)
	M = g.calculer_affichage_optimise()
	g.afficher_points(M)