import core
np = core.np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt

class GestionnaireAffichage():
	def __init__(self, G):
		self.G = G
		self.M = None

	def calculer_points_affichage(self):
		plt.clf()
		n = len(self.G)
		X = np.random.uniform(size=n)
		Y = np.random.uniform(size=n)
		return np.array([[x, y] for (x, y) in zip(X, Y)])

	def afficher_points_vecteurs(self, M, gradient):
		n = len(M)	
		X, Y = M[:, 0], M[:, 1]
		U, V = gradient[:, 0], gradient[:, 1]
		plt.quiver(X, Y, U, V)
		for i in range(n):
			for j in range(i, n):
				if self.G[i, j] == 1: 
					plt.plot([X[i], X[j]], [Y[i], Y[j]], color='k')
		plt.show()

	def calculer_affichage_optimise(self):
		D_star = calculer_D_star(floyd_warshall(G))
		M = self.calculer_points_affichage()
		n = len(M)
		gradient = calculer_gradient_energie(M, D_star)
		self.afficher_points_vecteurs(M, gradient) #juste pour tester

	def afficher_graphe(self, show=True):
		n = len(self.G)
		M = self.calculer_points_affichage()
		self.M = M
		X, Y = M[:, 0], M[:, 1]
		plt.plot(X, Y, 'bo')
		for i in range(n):
			for j in range(i, n):
				if self.G[i, j] == 1: 
					plt.plot([X[i], X[j]], [Y[i], Y[j]], color='k')
		if show:
			plt.show()


#Pas necessaire normalement
#def energie(M, D):
#	"M : points associes a chaque sommet du graphe, sous forme d'array numpy N x 2'\
#	D : distances entre les differents sommets du graphe, sous forme d'une matrice numpy"
#	N = len(M)
#	#on met des 1 sur la diagonale de D pour eviter la division par 0
#	D_star = D/np.max(D)
#	D_star2 = D_star + np.eye(N)
#	distances_M = np.array([[np.linalg.norm(M[i] - M[j]) for j in range(N)] for i in range(N)])
#	numerateur = (np.sqrt(0.5)*distances_M - D_star)
#	return np.sum(numerateur/D_star2)

def calculer_gradient_energie(M, D_star):
	"Renvoie une matrice N x 2 qui correspond au 'gradient' de l'energie par rapport \
	a chacune des composantes"
	tmp_vec = lambda vec, dist: [vec/dist, np.zeros((2,))][dist == 0.0]
	tmp_coeffs = lambda x, dist: [x/dist, 0.0][dist == 0.0]
	#ces deux fonctions permettent d'effectuer les operations avec numpy sans avoir des nan partout

	n = len(M)
	distances_M = np.array([[np.linalg.norm(M[i] - M[j]) for j in range(n)] for i in range(n)])

	vecteurs_M  = np.array([[tmp_vec((M[i] - M[j]), distances_M[i, j]) for j in range(n)] for i in range(n)])
	#vecteurs_= vecteurs directeurs des aretes entre chacun des points
	coefficients = np.sqrt(0.5)*np.array([[tmp_coeffs(np.sqrt(0.5)*distances_M[i, j] - D_star[i, j], D_star[i, j]) \
		for i in range(n)] for j in range(n)]) #tableau de taille (N, N)
	gradient_E = np.dot(coefficients, vecteurs_M)
	gradient_E = np.array([gradient_E[i, i] for i in range(n)])
	return gradient_E

def floyd_warshall(G):
	M, n = np.copy(G), len(G)
	f = lambda x: [float('inf'), 1.0][int(x)]
	M = np.array([[f(M[i, j]) for j in range(n)] for i in range(n)])
	for i in range(n):
		M[i, i] = 0.0
	for k in range(n):
		for i in range(n):
			for j in range(n):
				M[i, j] = min(M[i, j], M[i, k] + M[k, j])
	return M #M est la matrice de distance

def calculer_D_star(D):
	D_max = np.max(D)
	return D/D_max

if __name__ == '__main__':
	G = core.construire_G(5)
	g = GestionnaireAffichage(G)
	g.calculer_affichage_optimise()