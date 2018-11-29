import core
np = core.np
import matplotlib.pyplot as plt
import scipy.stats as stats

"""
Nota bene : si on a un linspace l , et qu'on veut selectionner une sous matrice a partir de l
(i.e les colonnes dans l et lignes dans l), on fait M[l][:, l]
"""

class GestionnaireCalcul:
	#TODO : Implémenter la possibilité de continuer les simulations de Monte-Carlo
	def __init__(self, N, M):
		self.M, self.N = M, N
		self.sommets_degres_comptes = None
	
	def estimation_sdc(self, verbose=False):
		N, M = self.N, self.M
		sommets_degres_comptes = np.zeros((N, N), dtype=int)
		for _ in range(M):
			if verbose and _ % 10 == 0:
				print("Nombre de graphes calcules : ", _)
			G = core.construire_G(N)
			degres = np.sum(G, axis=0).astype(int)  #ici, on peut sommer selon l'axe 0 ou 1
			#puisque la matrice est symetrique
			for i in range(N):
				d = degres[i]
				sommets_degres_comptes[i, (d-1)] += 1
		#degres_sommets[i, j] = nombre de fois que le sommet i a le degre j au total
		#on l'enregistre en memoire 
		self.sommets_degres_comptes = sommets_degres_comptes
	
	#todo : calculer la proba des degres
	def calculer_probas(self):
		#1) on calcule, pour chaque sommet, la proba que ce sommet soit d'un degre k :
		#on divise simplement par M
		try:
			probas_degres_sommets = self.sommets_degres_comptes/float(self.M)
			#on fait la moyenne sur les sommets qui sont les indices des lignes de la matrice
			#donc on fait np.average(..., axis=0)
			probas_degres = np.average(probas_degres_sommets, axis=0)
			return probas_degres
		except Exception as e:
			print("Erreur dans la matrice sommets_degres_comptes :", e)

	def charger_sdc(self, fichier, M):
		self.sommets_degres_comptes = np.load(fichier)
		self.M = M

	def sauvegarder_sdc(self, fichier):
		np.save(fichier, self.sommets_degres_comptes)

def afficher_log_log(probas,M=0):
	N = len(probas)
	X = np.log(np.linspace(1, N, N))
	Y = np.log(probas)
	slope, intercept, rvalue, pvalue, std_err = stats.linregress(X, y=Y)
	print("Coefficient directeur de la regression : ", slope)
	plt.plot(X, Y)
	Y_droite = X*slope + intercept
	plt.plot(X, Y_droite)
	plt.suptitle("Courbe de log(P(s = d)) en fonction de log(d). N = " +str(N) + ", M = " + str(M))
	plt.show()

if __name__ == '__main__':	
	N, M = 5000, 500
	gc = GestionnaireCalcul(N, M)
	gc.estimation_sdc(verbose=True)
	probas = gc.calculer_probas()
	probas = probas[:np.where(probas <= 0.0)[0][0]]
	afficher_log_log(probas, M)
	#np.save(probas)