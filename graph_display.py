import core
np = core.np
import matplotlib.pyplot as plt

def GestionnaireAffichage():
	def __init__(self, G):
		self.G = G
		self.M = None

	def calculer_points_affichage(self):
		plt.clf()
		n = len(self.G)
		X = np.random.uniform(size=n)
		Y = np.random.uniform(size=n)
		return np.array([[x, y] for (x, y) in zip(X, Y)])

	def afficher_graphe(self):
		M = self.calculer_points_affichage()
		X, Y = M[:, 0], M[:, 1]
		plt.plot(X, Y, 'bo')
		for i in range(n):
			for j in range(i, n):
				if self.G[i, j] == 1: 
					plt.plot([X[i], X[j]], [Y[i], Y[j]], color='k')
		plt.show()

def energie(M, D):
	"M : points associes a chaque sommet du graphe, sous forme d'array numpy N x 2'\
	D : distances entre les differents sommets du graphe, sous forme d'une matrice numpy"
	N = len(M)
	#on met des 1 sur la diagonale de D pour eviter la division par 0
	D_star = D/np.max(D)
	D_star2 = D_star + np.eye(N)
	distances_M = np.array([[np.linalg.norm(M[i] - M[j]) for j in range(N)] for i in range(N)])
	numerateur = (np.sqrt(0.5)*distances_M - D_star)
	return np.sum(numerateur/D_star2)

def floyd_warshall(G):
	M, n = np.copy(G), len(G)	
	for k in range(n):
		for i in range(n):
			for j in range(n):
				pass #TODO 

if __name__ == '__main__':
	G = core.construire_G(10)
	display_graph(G)