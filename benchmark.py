import core, time
import numpy as np

def compare_construire_G(size, nb_runs):
	times_construire_1 = np.zeros(nb_runs)
	for i in range(nb_runs):
		start = time.time()
		core.construire_G(size)
		end = time.time()
		times_construire_1[i] = end - start
	avg_1 = np.average(times_construire_1)
	times_construire_2 = np.zeros(nb_runs)
	for i in range(nb_runs):
		start = time.time()
		core.construire_G_optimise(size)
		end = time.time()
		times_construire_2[i] = end - start
	avg_2 = np.average(times_construire_2)
	return avg_1, avg_2

print(compare_construire_G(500, 100))
#Avec ces parametres : size = 500, nb_runs = 100 : 
#non optimise : 0.20192468643188477
#optimise : 0.10074010848999024