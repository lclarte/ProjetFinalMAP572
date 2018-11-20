import core
np = core.np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt

class GestionnaireAffichage():
	def __init__(self, G):
		self.G = G
		n = len(self.G)
		self.M = None
		self.nb_iter = 1000
		self.delta_t = 0.01
		self.seuil   = 0.0001 #norme du gradient de E "par sommet"
		self.suptitle = "Affichage optimise pour " + str(n) + " points. \n Nombre d'iterations : " + str(self.nb_iter) + \
				"; dt : " + str(self.delta_t)

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
		plt.suptitle(self.suptitle)
		plt.show()

	def calculer_affichage_optimise(self, verbose=True):
		#initialisation
		D_star = calculer_D_star(floyd_warshall(self.G))
		M = self.calculer_points_affichage()
		n = len(M)
		gradient = None
		energies = []
		grad_e_normes = []
		#on essaie avec un certain nombre d'iterations 
		while gradient is None or np.linalg.norm(gradient) >= n*n*self.seuil:
		#for iter in range(self.nb_iter):
			gradient = calculer_gradient_energie(M, D_star)
			M += -self.delta_t*gradient
			energies.append(energie(M, D_star))
			grad_e_normes.append(np.linalg.norm(gradient))
		if verbose:
			plt.plot(np.linspace(1, len(energies), len(energies)), grad_e_normes)
			plt.suptitle("Evolution du gradient de l'energie en fonction des iterations")
			plt.show()
		return M

def energie(M, D_star):
	n = len(M)
	distances = np.array([[np.linalg.norm(M[i] - M[j]) for j in range(n)] for i in range(n)])
	numerateur = (np.sqrt(0.5)*distances - D_star)**2
	tmp_f = lambda x : [x, 1.0][x == 0.0]
	denominateur = np.array([[tmp_f(D_star[i, j]**2) for i in range(n)] for j in range(n)])
	return np.sum(numerateur/denominateur)

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
	M = g.calculer_points_affichage()
	g.afficher_points(M, debug=False)
	M = g.calculer_affichage_optimise()
	g.afficher_points(M, debug=False, D=floyd_warshall(g.G))