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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score as NMI

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

def sim_to_dist(sim):
    out = np.zeros((len(sim),len(sim)))
    for i in range(len((sim))):
        for j in range(len(sim)):
            out[i][j] = (1 - sim[i][j])
    return(out)

def test_clustering(data, Y, coltype):
    nmis = []
    times = []
    n_clusters = len(set(Y))
    for i in range(20):
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed",linkage="average")
        start = time.time()
        sim = build_ensemble_inc(data, coltypes=coltype)
        duration = time.time() - start
        times.append(duration)
        distance = sim_to_dist(sim)
        predicted = cluster.fit_predict(distance)
        nmis.append(NMI(Y, predicted))
    print("Summary of all the runs.\n")
    print("Mean nmi : {0} (standard deviation : {1}), mean duration : {2} (standard deviation : {3}) \n".format(np.mean(nmis),np.std(nmis),np.mean(duration), np.std(duration)))
    print("----------\n")

print("-----------------------------------------------------\n")
print("Mixed types datasets\n")
print("-----------------------------------------------------\n")

print("----------")
print("Synthetic datasets")
print("----------")

a = np.random.uniform(0,0.5,250)
a = np.append(a,np.random.uniform(0.5,1,250))
b = np.random.uniform(1,2,250)
b = np.append(b,np.random.uniform(0,1,250))
c = []
d = []
e = []
f = []
dist = multinomial(250,[0.8,0.2,0.2])
for index,value in enumerate(dist):
	c.extend([index+1]*value)
dist = multinomial(250,[0.2,0.2,0.8])
for index,value in enumerate(dist):
	c.extend([index+1]*value)

dist = multinomial(250,[0.3,0.1,0.6])
for index,value in enumerate(dist):
	d.extend([index+1]*value)
dist = multinomial(250,[0.8,0.15,0.05])
for index,value in enumerate(dist):
	d.extend([index+1]*value)

dist = multinomial(250,[0.9,0.05,0.05])
for index,value in enumerate(dist):
	e.extend([index+1]*value)
dist = multinomial(250,[0.1,0.15,0.75])
for index,value in enumerate(dist):
	e.extend([index+1]*value)

dist = multinomial(250,[0.7,0.1,0.05])
for index,value in enumerate(dist):
	f.extend([index+1]*value)
dist = multinomial(250,[0.1,0.8,0.1])
for index,value in enumerate(dist):
	f.extend([index+1]*value)

g = np.random.uniform(0,0.2,250)
g = np.append(g,np.random.uniform(0.4,0.6,250))
h = np.random.uniform(4,5,250)
h = np.append(h, np.random.uniform(0,1,250))
data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f,g,h)))
cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[0,0,1,1,1,1,0,0])
print(cluster_corr[2])
print("Test dataset (with clusters - mixed values) - 500 samples. 4 continuous columns, 4 categorical column with 3 modalities")
test_clustering(*cluster_corr)

del cluster_corr

print("-----------------------------------------------------\n")
print("Categorical datasets\n")
print("-----------------------------------------------------\n")

print("----------")
print("Synthetic datasets")
print("----------")

a = []
dist = multinomial(250,[0.8,0.2,0.2])
for index,value in enumerate(dist):
	a.extend([index+1]*value)
dist = multinomial(250,[0.2,0.2,0.8])
for index,value in enumerate(dist):
	a.extend([index+1]*value)

b = []

dist = multinomial(250,[0.7,0.1,0.05])
for index,value in enumerate(dist):
	b.extend([index+1]*value)
dist = multinomial(250,[0.1,0.8,0.1])
for index,value in enumerate(dist):
	b.extend([index+1]*value)

data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b)))
cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[1,1])
print(cluster_corr[2])
print("Test dataset (with clusters - Categorical) - 500 samples. 2 categorical columns, 2 classes")
test_clustering(*cluster_corr)

del cluster_corr


print("----------")
print("Iris dataset")
print("----------")

data_iris = datasets.load_iris()
iris_type = [0 for i in range(4)]
iris = (np.array(data_iris.data), np.array(data_iris.target),iris_type)
print("Iris dataset")
test_clustering(*iris)
del iris


print("----------")
print("Wisconsin dataset")
print("----------")

data_wisc = pd.read_csv("./donnees/wisconsin.txt",sep="\t",header=None).astype('float64')
wisc_type = [0 for i in range(len(data_wisc.columns))]
wisconsin = (data_wisc.iloc[:,:-1].values,data_wisc.iloc[:,-1].values,wisc_type)
print("Wisconsin dataset")
test_clustering(*wisconsin)
del wisconsin

print("----------")
print("Soybean dataset")
print("----------")

data_soybean = pd.read_csv("./donnees/soybean.data",sep="\t",header=None).astype('float64')
soybean_type = [1 for i in range(len(data_soybean.columns))]
soybean = (data_soybean.iloc[:,:-1].values, data_soybean.iloc[:,-1].values,soybean_type)
print("Soybean dataset")
test_clustering(*soybean)
del soybean