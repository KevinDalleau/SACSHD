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
from numpy.random import multinomial

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

print("-----------------------------------------------------\n")
print("Continuous datasets\n")
print("-----------------------------------------------------\n")

print("----------")
print("Synthetic datasets")
print("----------")

a = np.random.uniform(0,0.5,250)
a = np.append(a,np.random.uniform(0.5,1,250))
b = np.random.uniform(1,2,250)
b = np.append(b,np.random.uniform(0,1,250))
data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b)))
cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[0,0])
print(cluster_corr[2])
print("Test dataset (with clusters - Continuous) - 500 samples. 2 continuous columns, 2 classes")
test_differences(*cluster_corr)

del cluster_corr

a = np.random.uniform(0,0.5,150)
a = np.append(a,np.random.uniform(0.5,1,150))
a = np.append(a,np.random.uniform(1,1.5,150))

b = np.random.uniform(1,2,150)
b = np.append(b,np.random.uniform(0,1,150))
b = np.append(b,np.random.uniform(2,3,150))

data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b)))
cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 150 else 1 if i < 300 else 2 for i in range(len(data_cluster_corr))],[0,0])
print(cluster_corr[2])
print("Test dataset (with clusters - Continuous) - 450 samples. 2 continuous columns, 3 classes")
test_differences(*cluster_corr)

del cluster_corr


a = np.random.uniform(0,0.5,250)
a = np.append(a,np.random.uniform(0.5,1,250))
b = np.random.uniform(1,2,250)
b = np.append(b,np.random.uniform(0,1,250))

c = np.random.uniform(0,0.2,250)
c = np.append(c,np.random.uniform(0.4,0.6,250))
d = np.random.uniform(4,5,250)
d = np.append(d, np.random.uniform(0,1,250))
data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d)))
cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[0,0,0,0])
print(cluster_corr[2])
print("Test dataset (with clusters - Continuous) - 500 samples. 4 continuous columns")
test_differences(*cluster_corr)

del cluster_corr

