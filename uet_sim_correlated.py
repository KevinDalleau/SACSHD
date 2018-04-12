import numpy as np
import pandas as pd
import random
import math
from joblib import Parallel, delayed
import multiprocessing
from sklearn import preprocessing
import time
from sklearn import datasets
import uetlib
from scipy.stats import ttest_ind, ttest_rel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_classification, make_blobs

def build_ensemble(data,n_estimators=50,nmin=None,coltypes=None):
	if nmin == None:
		nmin = math.floor(len(data)/3)
	similarities = []
	num_cores = multiprocessing.cpu_count()
	results = Parallel(n_jobs=num_cores)(delayed(uetlib.get_sim)(data,nmin,coltypes) for i in range(n_estimators))
	similarities = results
	return(np.sum(similarities,axis=0)/n_estimators)

def build_ensemble_inc(data,n_estimators=50,nmin=None,coltypes=None):
	if nmin == None:
		nmin = math.floor(len(data)/3)
	similarities = np.zeros((len(data),len(data)))
	num_cores = multiprocessing.cpu_count()
	results = Parallel(n_jobs=num_cores)(delayed(uetlib.get_sim_one)(data,nmin,coltypes) for i in range(n_estimators))

	similarities = results
	
	return(np.sum(similarities,axis=0)/n_estimators)

def compute_distance(similarity):
	out = [[0 for i in range(len(similarity))] for j in range(len(similarity))]
	for indexi,i in enumerate(similarity):
		for indexj,j in enumerate(similarity):
			out[indexi][indexj] = math.sqrt(1-similarity[indexi][indexj])
	return(np.array(out))

def compute_sim_intra_inter(similarity,classes):
	distinct_classes = np.unique(classes)
	indices = []
	for current in distinct_classes:
		indices.append(np.where(classes==current)[0])
	intrac_similarities = []
	interc_similarities = []
	for i in range(len(indices)):
		cluster1_indices = indices[i]
		for j in range(len(indices)):
			cluster2_indices = indices[j]
			if(i==j):
				local_sim = [similarity[i][j] for i in cluster1_indices for j in cluster2_indices if (i != j) ]
				intrac_similarities.extend(local_sim)
			else:
				local_sim = [similarity[i][j]  for i in cluster1_indices for j in cluster2_indices if (i != j)]
				interc_similarities.extend(local_sim)
	return((intrac_similarities, interc_similarities))

def test_differences(data,Y,coltype):
	global_difference = []
	for i in range(20):
		matrices = []
		for j in range(5):
			similarity = build_ensemble_inc(data, coltypes=coltype)
			matrices.append(similarity)
		similarity_matrix = np.sum(matrices, axis = 0)/5
		sims = compute_sim_intra_inter(similarity_matrix, Y)
		intrac = sims[0]
		interc = sims[1]
		intrac_mean = np.mean(intrac)
		interc_mean = np.mean(interc)
		global_difference.append(abs(interc_mean-intrac_mean))
	print("Summary of all the runs.\n")
	print("Mean difference between intercluster similarities and intracluster similarities : {0} (standard deviation : {1}) \n".format(np.mean(global_difference),np.std(global_difference)))
	print("----------\n")
	return((np.mean(global_difference),np.std(global_difference)))



# out = []
# for n in [i for i in range(0,6)]:
# 	local_out = ()
# 	n_informative = 10 - n
# 	n_redundant = n
# 	X, y = make_classification(n_samples=200, n_features=10,n_informative = n_informative, n_redundant =n_redundant,random_state=0, flip_y =0,n_classes = 2, n_clusters_per_class=1, shuffle = False, class_sep = 2)
# 	data = X
# 	data_type = [0 for i in range(data.shape[1])]

# 	blob = (np.array(data), np.array(y),data_type)
# 	print("Synthetic dataset, 200 samples, 10 features. Number of redundant features = {0}".format(n_redundant))
# 	diff = test_differences(*blob)

# 	local_out = diff + (n_redundant,)
# 	out.append(local_out)
# 	del blob

# print(out)
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# x_ax = [value[2] for value in out]

# noise_mean = [value[0] for value in out]
# noise_std = [value[1] for value in out]
# plt.ylim(0,1)

# plt.errorbar(x_ax,noise_mean,yerr = noise_std, label = "Classification")

# plt.xlabel("Number of redundant features")
# plt.ylabel("Mean difference between intracluster and intercluster similarities")
# plt.savefig("correlation_classification.png")
# plt.clf()
# plt.cla()
# plt.close()

out = []
for n in [i for i in range(0,10)]:
	local_out = ()
	n_informative = 10 - n
	n_redundant = n
	X, y = make_blobs(n_samples=500, centers=3, n_features=10,random_state=0)
	data = X
	data_type = [0 for i in range(data.shape[1])]
	if (n_redundant == 1):
		data[:,-1] = np.random.rand(data.shape[0])
	if (n_redundant ==  2):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
	if (n_redundant ==  3):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
	if (n_redundant ==  4):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
		data[:,-4] = np.random.rand(data.shape[0])
	if (n_redundant ==  5):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
		data[:,-4] = np.random.rand(data.shape[0])
		data[:,-5] = np.random.rand(data.shape[0])
	
	if (n_redundant ==  6):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
		data[:,-4] = np.random.rand(data.shape[0])
		data[:,-5] = np.random.rand(data.shape[0])
		data[:,-6] = np.random.rand(data.shape[0])

	if (n_redundant ==  7):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
		data[:,-4] = np.random.rand(data.shape[0])
		data[:,-5] = np.random.rand(data.shape[0])
		data[:,-6] = np.random.rand(data.shape[0])
		data[:,-7] = np.random.rand(data.shape[0])

	if (n_redundant ==  8):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
		data[:,-4] = np.random.rand(data.shape[0])
		data[:,-5] = np.random.rand(data.shape[0])
		data[:,-6] = np.random.rand(data.shape[0])
		data[:,-7] = np.random.rand(data.shape[0])
		data[:,-8] = np.random.rand(data.shape[0])

	if (n_redundant ==  9):
		data[:,-1] = np.random.rand(data.shape[0])
		data[:,-2] = np.random.rand(data.shape[0])
		data[:,-3] = np.random.rand(data.shape[0])
		data[:,-4] = np.random.rand(data.shape[0])
		data[:,-5] = np.random.rand(data.shape[0])
		data[:,-6] = np.random.rand(data.shape[0])
		data[:,-7] = np.random.rand(data.shape[0])
		data[:,-8] = np.random.rand(data.shape[0])
		data[:,-9] = np.random.rand(data.shape[0])


	
	print(data[0])
	blob = (np.array(data), np.array(y),data_type)
	print("Synthetic dataset, 500 samples, 10 features. Number of redundant features = {0}".format(n_redundant))
	diff = test_differences(*blob)


	local_out = diff + (n_redundant,)
	out.append(local_out)
	del blob

print(out)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
x_ax = [value[2] for value in out]

noise_mean = [value[0] for value in out]
noise_std = [value[1] for value in out]
plt.ylim(0,1)

plt.errorbar(x_ax,noise_mean,yerr = noise_std, label = "Random values")


out = []
for n in [i for i in range(0,10)]:
	local_out = ()
	n_informative = 10 - n
	n_redundant = n
	X, y = make_blobs(n_samples=500, centers=3, n_features=10,random_state=0)
	data = X
	data_type = [0 for i in range(data.shape[1])]
	if (n_redundant == 1):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
	if (n_redundant ==  2):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
	if (n_redundant ==  3):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
	if (n_redundant ==  4):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-4] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
	if (n_redundant ==  5):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-4] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)

	if (n_redundant ==  6):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-4] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-6] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)


	if (n_redundant ==  7):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-4] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-6] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-7] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)

	if (n_redundant ==  8):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-4] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-6] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-7] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-8] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)

	if (n_redundant ==  9):
		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-2] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-4] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-6] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-7] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-8] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
		data[:,-9] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)


	
	print(data[0])
	blob = (np.array(data), np.array(y),data_type)
	print("Synthetic dataset, 500 samples, 10 features. Number of redundant features = {0}".format(n_redundant))
	diff = test_differences(*blob)


	local_out = diff + (n_redundant,)
	out.append(local_out)
	del blob

print(out)

x_ax = [value[2] for value in out]

noise_mean = [value[0] for value in out]
noise_std = [value[1] for value in out]


plt.errorbar(x_ax,noise_mean,yerr = noise_std, label = "Correlated values of one variable")


# out = []
# for n in [i for i in range(0,10)]:
# 	local_out = ()
# 	n_informative = 10 - n
# 	n_redundant = n
# 	X, y = make_blobs(n_samples=500, centers=3, n_features=10,random_state=0)
# 	data = X
# 	data_type = [0 for i in range(data.shape[1])]
# 	if (n_redundant == 1):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 	if (n_redundant ==  2):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 	if (n_redundant ==  3):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 	if (n_redundant ==  4):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-4] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 	if (n_redundant ==  5):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-4] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)

# 	if (n_redundant ==  6):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-4] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-6] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)


# 	if (n_redundant ==  7):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-4] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-6] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-7] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)

# 	if (n_redundant ==  8):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-4] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-6] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-7] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-8] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)

# 	if (n_redundant ==  9):
# 		data[:,-1] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-2] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-3] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-4] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-5] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-6] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-7] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)
# 		data[:,-8] = np.random.uniform(0,100)*data[:,1] + np.random.uniform(0,100)
# 		data[:,-9] = np.random.uniform(0,100)*data[:,0] + np.random.uniform(0,100)


	
# 	print(data[0])
# 	blob = (np.array(data), np.array(y),data_type)
# 	print("Synthetic dataset, 500 samples, 10 features. Number of redundant features = {0}".format(n_redundant))
# 	diff = test_differences(*blob)


# 	local_out = diff + (n_redundant,)
# 	out.append(local_out)
# 	del blob

# print(out)

# x_ax = [value[2] for value in out]

# noise_mean = [value[0] for value in out]
# noise_std = [value[1] for value in out]


# plt.errorbar(x_ax,noise_mean,yerr = noise_std, label = "Correlated values of two variables")

plt.legend()
plt.xlabel("Number of redundant features")
plt.ylabel("Mean difference between intracluster and intercluster similarities")
plt.savefig("correlation_blobs_random_and_correlated.png")
plt.clf()
plt.cla()
plt.close()

