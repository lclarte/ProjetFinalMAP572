import core
import graph_display as g_d
import numpy as np
from clustering import *

k = 20
G = np.loadtxt("StochasticBlockModel.txt")
cm = ClusteringManager(G, k)
cm.calculer_clustering_G()