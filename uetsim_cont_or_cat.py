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


print("----------")
print("Iris dataset")
print("----------")

data_iris = datasets.load_iris()
iris_type = [0 for i in range(4)]
iris = (np.array(data_iris.data), np.array(data_iris.target),iris_type)
print("Iris dataset")
test_differences(*iris)
del iris

print("----------")
print("Wisconsin dataset")
print("----------")

data_wisc = pd.read_csv("./donnees/wisconsin.txt",sep="\t",header=None).astype('float64')
wisc_type = [0 for i in range(len(data_wisc.columns))]
wisconsin = (data_wisc.iloc[:,:-1].values,data_wisc.iloc[:,-1].values,wisc_type)
print("Wisconsin dataset")
test_differences(*wisconsin)
del wisconsin

print("----------")
print("Promoters dataset")
print("----------")

data_promoters = pd.read_csv("./donnees/promoters.txt",sep="\t",header=None).astype('float64')
promoters_type = [1 for i in range(len(data_promoters.columns))]
promoters = (data_promoters.iloc[:,:-1].values, data_promoters.iloc[:,-1].values,promoters_type)
print("Promoters dataset")
test_differences(*promoters)
del promoters

print("----------")
print("Soybean dataset")
print("----------")

data_soybean = pd.read_csv("./donnees/soybean.data",sep="\t",header=None).astype('float64')
soybean_type = [1 for i in range(len(data_soybean.columns))]
soybean = (data_soybean.iloc[:,:-1].values, data_soybean.iloc[:,-1].values,soybean_type)
print("Soybean dataset")
test_differences(*soybean)
del soybean

print("----------")
print("Spect dataset")
print("----------")

data_spect = pd.read_csv("./donnees/spect.txt",sep="\t",header=None).astype('float64')
spect_type = [1 for i in range(len(data_spect.columns))]
spect = (data_spect.iloc[:,1:].values, data_spect.iloc[:,0].values,spect_type)
print("Spect dataset")
test_differences(*spect)
del spect


print("----------")
print("Madelon dataset")
print("----------")

data_madelon = pd.read_csv("./donnees/madelon.txt",sep="\t",header=None).astype('float64')
madelon_type = [0 for i in range(len(data_madelon.columns))]
madelon = (data_madelon.iloc[:,:-1].values, data_madelon.iloc[:,-1].values,madelon_type)
print("Madelon dataset")
test_differences(*madelon)
del madelon

print("----------")
print("Isolet dataset")
print("----------")

data_isolet  = pd.read_csv('./donnees/isolet.txt',sep="\t",header=None).astype('float64')
isolet_type = [0 for i in range(len(data_isolet.columns))]
isolet = (data_isolet.iloc[:,:-1].values, data_isolet.iloc[:,-1], isolet_type)
print("Isolet dataset")
test_differences(*isolet)
del isolet

print("----------")
print("Pima dataset")
print("----------")

data_pima = pd.read_csv('./donnees/Pima.txt',sep="\t",header=None).astype('float64')
pima_type = [0 for i in range(len(data_pima.columns))]
pima = (data_pima.iloc[:,:-1].values, data_pima.iloc[:,-1].values,pima_type)
print("Pima dataset")
test_differences(*pima)
del pima

print("----------")
print("Spam base dataset")
print("----------")

data_spam = pd.read_csv("./donnees/spamb.txt",sep="\t",header=None).astype('float64')
spam_type = [0 for i in range(len(data_spam.columns))]
spam = (data_spam.iloc[:,:-1].values,data_spam.iloc[:,-1].values, spam_type)
print("Spam base dataset")
test_differences(*spam)
del spam
