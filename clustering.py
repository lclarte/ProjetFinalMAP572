import numpy as np
import scipy.linalg as linalg

class ClusteringManager():
	def __init__(self, G, k):
		self.G = G
		self.k = k