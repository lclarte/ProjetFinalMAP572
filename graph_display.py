import core, time
np = core.np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opti
import scipy.sparse.csgraph as csgraph

class GestionnaireAffichage:
	def __init__(self, G):
		self.G = G
		n = len(self.G)
		self.M = None
		self.afficher_aretes = True
		self.nb_iter = 1000
		self.options = {'maxiter': self.nb_iter}
		self.delta_t = 0.01
		self.seuil   = 0.01 #norme du gradient de E "par sommet"
		self.suptitle = ""
		self.verbose = False #si on detaille lors de la descente de gradient
		self.verbose_graphe = False #si on afficher le graphe de la norme du gradient apres le calcul
		self.fonction_gradient = self.affichage_optimise_gradient
		self.afficher_pts_bool = True
		self.proba_afficher_arete = 1.0

	def calculer_points_affichage(self):
		n = len(self.G)
		X = np.random.uniform(size=n)
		Y = np.random.uniform(size=n)
		return np.array([[x, y] for (x, y) in zip(X, Y)])

	#Fonction un peu inutile actuellement, sert juste a visualiser les vecteurs 
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
		if self.afficher_pts_bool:
			for i in range(n):
				#plt.text(X[i], Y[i], str(i+1))
				plt.scatter(X[i], Y[i],c=colors[labels[i]])
		for i in range(n):
			for j in range(n):
				if self.afficher_aretes and self.G[i, j] == 1 and np.random.uniform() <= self.proba_afficher_arete:
					plt.plot([X[i], X[j]], [Y[i], Y[j]], linestyle=':', color='k', marker=",")
		plt.suptitle(self.suptitle)
		plt.show()

	def calculer_affichage_optimise(self, method=0):
		"Si method = 0, on utilise notre methode personnelle"
		#initialisation
		#ancienne version : D_star = calculer_D_star(floyd_warshall(self.G))
		D_star = calculer_D_star(csgraph.floyd_warshall(self.G))	 
		M = self.calculer_points_affichage()
		n = len(M)
		grad_e_normes, energies = [], []
		if method == 0:
			M, grad_e_normes, energies = self.fonction_gradient(M, D_star, self.verbose)
			if self.verbose_graphe:
				plt.plot(np.linspace(1, len(energies), len(energies)), grad_e_normes)
				plt.suptitle("Evolution du gradient de l'energie en fonction des iterations")
				plt.show()
		elif method == 1:
			print("M.shape = ", M.shape)
			res = opti.minimize(lambda m: energie_vec(m, D_star), vectoriser_M(M), options = self.options, method='CG') #on utilise la methode du gradient conjugue pour ameliorer la vitesse
			M = matriciser_M(res.x)
			print(res.success)
			if not res.success:
				raise Exception("La minimisation n'a pas convergé")
		else:
			raise Exception("Argument method incorrect")
		return M

	def affichage_optimise_gradient(self, M, D_star, verbose=False):
		n = len(M)
		gradient = None
		grad_e_normes = []
		energies = []
		it = 0
		depart = time.time()
		while gradient is None or (np.linalg.norm(gradient)**2 >= n*self.seuil and it < self.nb_iter):
			it += 1
			if verbose:
				#print("iteration numero ", it)
				#print("Temps ecoule depuis la derniere iteration : ", time.time() - depart)
				depart = time.time()
			gradient = calculer_gradient_energie(M, D_star)
			M += -self.delta_t*gradient
			energies.append(energie(M, D_star))
			grad_e_normes.append(np.linalg.norm(gradient))
		if verbose:
			if np.linalg.norm(gradient)**2 < n*self.seuil:
				print("Le seuil a ete atteint !")
			if it >= self.nb_iter:
				print("Le nombre max d'iterations a ete depasse")

		return M, grad_e_normes, energies

	def tester_vitesse_iteration(self):
		M = 10
		essais = np.zeros(M)
		for i in range(M):
			M = self.calculer_points_affichage()
			D_star = calculer_D_star(csgraph.floyd_warshall(self.G))
			depart = time.time()
			calculer_gradient_energie(M, D_star)
			essais[i] = time.time() - depart		
		return np.average(essais)

	def affichage_optimise_gradient_momentum(self, M, D_star):
		"Methode qui enregistre le moment precedent pour mettre a jour les coefficients. Ne semble pas bien fonctionner"
		gamma = 0.9
		n = len(M)
		gradient = None
		grad_e_normes = []
		vitesse  = np.zeros(M.shape)
		energies = []
		while gradient is None or np.linalg.norm(gradient) >= n*n*self.seuil:
			gradient = calculer_gradient_energie(M, D_star)
			vitesse = gamma	* vitesse + self.delta_t*gradient
			M += - vitesse
			energies.append(energie(M, D_star))
			grad_e_normes.append(np.linalg.norm(gradient))
		return M, grad_e_normes, energies


class GestionnaireAffichage3D(GestionnaireAffichage):
	def __init__(self, G):
		super(GestionnaireAffichage3D, self).__init__(G)
		self.nb_iter = 1000
		self.options = {'maxiter': self.nb_iter}

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
				if self.afficher_aretes and self.G[i, j] == 1:
					ax.plot(X[[i, j]], Y[[i, j]], Z[[i, j]])
		plt.show()

	def calculer_affichage_optimise_0(self):
		#anciennce version : D_star = calculer_D_star(floyd_warshall(self.G))
		D_star = calculer_D_star(csgraph.floyd_warshall(self.G))
		M = self.calculer_points_affichage()
		res = opti.minimize(lambda m:energie_vec(m, D_star, dim=3), vectoriser_M(M), options = self.options, method='CG')
		M = matriciser_M(res.x, dim=3)
		return M

	def calculer_affichage_optimise(self, method=0):
		D_star = calculer_D_star(csgraph.floyd_warshall(self.G))	 
		M = self.calculer_points_affichage()
		n = len(M)
		grad_e_normes, energies = [], []
		if method == 0:
			M, grad_e_normes, energies = self.fonction_gradient(M, D_star, self.verbose)
			if self.verbose_graphe:
				plt.plot(np.linspace(1, len(energies), len(energies)), grad_e_normes)
				plt.suptitle("Evolution du gradient de l'energie en fonction des iterations")
				plt.show()
		elif method == 1:
			print("M.shape = ", M.shape)
			res = opti.minimize(lambda m: energie_vec(m, D_star, dim=3), vectoriser_M(M), options = self.options, method='CG') #on utilise la methode du gradient conjugue pour ameliorer la vitesse
			M = matriciser_M(res.x, dim=3)
			print(res.success)
			if not res.success:
				raise Exception("La minimisation n'a pas convergé")
		else:
			raise Exception("Argument method incorrect")
		return M

#FONCTIONS UTILISEES POUR FAIRE USAGE DE OPTIMIZE.MINIMIZE
def matriciser_M(M_vec, dim=2):
	n = int(len(M_vec)/dim)
	M = np.zeros((n, dim))
	for i in range(n):
		for j in range(dim):
			M[i, j] = M_vec[dim*i + j]
	return M
	
def vectoriser_M(M):
	n, dim = M.shape
	M_vec = np.zeros((dim*n,1))
	for i in range(n):
		for j in range(dim):
			 M_vec[dim*i + j] = M[i, j]
	return M_vec

#FONCTIONS UTILISEES POUR LE CALCUL DE L'ENERGIE PROPOSEE PAR L'ENONCE
def energie_vec(M_vec, D_star, dim=2):
	"Vectorisation de la fonction energie. Necessaire pour utiliser scipy.optimize.minimize"
	return energie(matriciser_M(M_vec, dim), D_star)

def energie(M, D_star):
	n = len(M)
	distances = np.array([[np.linalg.norm(M[i] - M[j]) for j in range(n)] for i in range(n)])
	numerateur = (np.sqrt(0.5)*distances - D_star)**2
	tmp_f = lambda x : [x, 1.0][x == 0.0]
	denominateur = np.array([[tmp_f(D_star[i, j]**2) for i in range(n)] for j in range(n)])
	return np.sum(numerateur/denominateur)

def calculer_gradient_energie(M, D_star):
	"Renvoie une matrice N x dim qui correspond au 'gradient' de l'energie par rapport \
	a chacune des composantes"
	dim = M.shape[1]
	tmp_vec = lambda vec, dist: [vec/dist, np.zeros((dim,))][dist == 0.0]
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

#ESSAI D'UNE AUTRE ENERGIE, BASEE SUR L'ENERGIE D'UN RESSORT
#NE MARCHE PAS ACTUELLEMENT
#ESSAYER UNE AUTRE METHODE : FAIRE LE BILAN DES FORCES SUR CHACUN DES SOMMETS COMME DANS LE MAILLAGE
def energie_ressorts_vec(M, G, dim=2):
	return energie_ressorts(matriciser_M(M, dim), G)

def energie_ressorts(M, G):
	"Proposition d'une autre energie, basee sur l'energie d'un ressort"
	n = len(M)
	epsilon = 0.5 #constance de raideur, arbitraire
	distances = np.array([[np.linalg.norm(M[i] - M[j]) for j in range(n)] for i in range(n)])
	distances_carre = distances**2
	energie_ressort = np.sum(distances_carre)
	distances_liens = ((distances - 1)**2)*G #on ne prend que les elements qui sont relies
	energie_liens = 0.5*np.sum(distances_liens) #on multiplie par 0.5 car on a pris les aretes deux fois en compte
	return - energie_ressort + 5*energie_liens

#Fonctions pour le calcul de la matrice D_star

#OBSOLETE : UTILISER A LA PLACE
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
