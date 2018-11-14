import core
np = core.np
import matplotlib.pyplot as plt

def display_graph(G):
	n = len(G)
	X = np.random.uniform(size=n)
	Y = np.random.uniform(size=n)
	plt.plot(X, Y, 'bo')
	for i in range(n):
		for j in range(i, n):
			if G[i, j] == 1: 
				plt.plot([X[i], X[j]], [Y[i], Y[j]], color='k')
	plt.show()

def improved_display_graph(G):
	pass

def floyd_warshall(G):
	M, n = np.copy(G), len(G)	
	for k in range(n):
		for i in range(n):
			for j in range(n):
				pass #TODO 

if __name__ == '__main__':
	G = core.construire_G(10)
	display_graph(G)