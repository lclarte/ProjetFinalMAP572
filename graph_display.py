import core
np = core.np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opti

class GestionnaireAffichage:
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

	def afficher_points(self, M, debug=True, D=None, labels=None, gradient=False):
		"gradient = True, on va utiliser des gradients de couleur pour l'affichage : \
		plus un sommet est bas, plus sa couleur sera foncee"
		assert labels is None or not gradient
		if labels is None:
			labels = [0]*len(M)
		colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'w']
		plt.clf()
		n = len(self.G)
		X, Y = M[:, 0], M[:, 1]
		if debug:
			if not D is None:
				print("distances : ", D)
		for i in range(n):
			plt.text(X[i], Y[i], str(i+1))
			plt.scatter(X[i], Y[i],c=colors[labels[i]])
		for i in range(n):
			for j in range(n):
				if self.G[i, j] == 1:
					plt.plot([X[i], X[j]], [Y[i], Y[j]], linestyle=':', color='k')
		plt.suptitle(self.suptitle)
		plt.show()

	def calculer_affichage_optimise(self, verbose=False, method=0):
		"Si method = 0, on utilise notre methode personnelle"
		#initialisation
		D_star = calculer_D_star(floyd_warshall(self.G))
		M = self.calculer_points_affichage()
		n = len(M)
		grad_e_normes, energies = [], []
		if method == 0:
			M, grad_e_normes, energies = self.affichage_optimise_gradient(M, D_star)
			if verbose:
				plt.plot(np.linspace(1, len(energies), len(energies)), grad_e_normes)
				plt.suptitle("Evolution du gradient de l'energie en fonction des iterations")
				plt.show()
		elif method == 1:
			res = opti.minimize(lambda m: energie_vec(m, D_star), vectoriser_M(M))
			M = matriciser_M(res.x)
			if not res.success:
				raise Exception("La minimisation n'a pas convergé")
		else:
			raise Exception("Argument method incorrect")
		return M

	def affichage_optimise_gradient(self, M, D_star):
		n = len(M)
		gradient = None
		grad_e_normes = []
		energies = []
		while gradient is None or np.linalg.norm(gradient) >= n*n*self.seuil:
		#for iter in range(self.nb_iter):
			gradient = calculer_gradient_energie(M, D_star)
			M += -self.delta_t*gradient
			energies.append(energie(M, D_star))
			grad_e_normes.append(np.linalg.norm(gradient))
		return M, grad_e_normes, energies


class GestionnaireAffichage3D(GestionnaireAffichage):
	def __init__(self, G):
		super(GestionnaireAffichage3D, self).__init__(G)

	def calculer_points_affichage(self):
		n = len(self.G)
		X = np.random.uniform(size=n)
		Y = np.random.uniform(size=n)
		Z = np.random.uniform(size=n)
		return np.array([[x, y, z] for (x, y, z) in zip(X, Y, Z)])

	def afficher_points(self, M):
		assert len(M) == len(self.G)
		n = len(M)
		fig = plt.figure()
		ax  = fig.add_subplot(111, projection='3d')
		X, Y, Z = M[:, 0], M[:, 1], M[:, 2]
		for i in range(n):
			ax.scatter([X[i]], [Y[i]], [Z[i]])
			for j in range(i, n):
				if self.G[i, j] == 1:
					ax.plot(X[[i, j]], Y[[i, j]], Z[[i, j]])
		plt.show()


def matriciser_M(M_vec):
	n = int(len(M_vec)/2)
	M = np.zeros((n, 2))
	for i in range(n):
		for j in range(2):
			M[i, j] = M_vec[2*i + j]
	return M

def vectoriser_M(M):
	n = len(M)
	M_vec = np.zeros((2*n,1))
	for i in range(n):
		for j in range(2):
			 M_vec[2*i + j] = M[i, j]
	return M_vec

def energie_vec(M_vec, D_star):
	"Vectorisation de la fonction energie. Necessaire pour utiliser scipy.optimize.minimize"
	return energie(matriciser_M(M_vec), D_star)

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

#Fonctions pour le calcul de la matrice D_star

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
