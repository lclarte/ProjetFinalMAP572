import core
np = core.np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt

class GestionnaireAffichage():
	def __init__(self, G):
		self.G = G
		self.M = None
		self.nb_iter = 1000
		self.delta_t = 0.01

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
					plt.plot([X[i], X[j]], [Y[i], Y[j]], color='r')
		plt.show()

	def afficher_points(self, M, debug=True, D=None):
		plt.clf()
		n = len(self.G)
		X, Y = M[:, 0], M[:, 1]
		if debug:
			if not D is None:
				print("distances : ", D)
			for i in range(n):
				plt.text(X[i], Y[i], str(i+1))
		for i in range(n):
			for j in range(n):
				if self.G[i, j] == 1:
					plt.plot([X[i], X[j]], [Y[i], Y[j]], color='k')
		plt.suptitle("Affichage optimise pour " + str(n) + " points. \n Nombre d'iterations : " + str(self.nb_iter) + \
				"; dt : " + str(self.delta_t))
		plt.show()

	def calculer_affichage_optimise(self):
		#initialisation
		D_star = calculer_D_star(floyd_warshall(G))
		M = self.calculer_points_affichage()
		n = len(M)
		#on essaie avec un certain nombre d'iterations 
		for iter in range(self.nb_iter):
			gradient = calculer_gradient_energie(M, D_star)
			M += -self.delta_t*gradient
		return M

def energie(M, D_star):
	pass#numerateur = np.sqrt(0.5)*

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
	G = core.construire_G(20)
	g = GestionnaireAffichage(G)
	M = g.calculer_affichage_optimise()
	g.afficher_points(M, debug=True, D=floyd_warshall(g.G))