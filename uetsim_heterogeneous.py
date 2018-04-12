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
test_differences(*cluster_corr)

del cluster_corr



a = np.random.uniform(0,0.5,250)
a = np.append(a,np.random.uniform(0.5,1,250))
b = np.random.uniform(1,2,250)
b = np.append(b,np.random.uniform(0,1,250))
c = []
d = []
e = []
f = []
# g = []
# h = []
# i = []
# j = []
# k = []
out = []
data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b)))
for n_columns in range(0,5):

	local_out = ()

	if n_columns == 1:
		dist = multinomial(250,[0.8,0.2,0.2])
		for index,value in enumerate(dist):
			c.extend([index+1]*value)
		dist = multinomial(250,[0.2,0.2,0.8])
		for index,value in enumerate(dist):
			c.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c)))

	if n_columns == 2:
		dist = multinomial(250,[0.3,0.1,0.6])
		for index,value in enumerate(dist):
			d.extend([index+1]*value)
		dist = multinomial(250,[0.8,0.15,0.05])
		for index,value in enumerate(dist):
			d.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d)))

	if n_columns == 3:
		dist = multinomial(250,[0.9,0.05,0.05])
		for index,value in enumerate(dist):
			e.extend([index+1]*value)
		dist = multinomial(250,[0.1,0.15,0.75])
		for index,value in enumerate(dist):
			e.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e)))

	if n_columns == 4:
		dist = multinomial(250,[0.7,0.1,0.05])
		for index,value in enumerate(dist):
			f.extend([index+1]*value)
		dist = multinomial(250,[0.1,0.8,0.1])
		for index,value in enumerate(dist):
			f.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f)))
		
	if n_columns == 5:
		dist = multinomial(250,[0.55,0.15,0.3])
		for index,value in enumerate(dist):
			g.extend([index+1]*value)
		dist = multinomial(250,[0.1,0.35,0.55])
		for index,value in enumerate(dist):
			g.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f,g)))
	
	if n_columns == 6:
		dist = multinomial(250,[0.2,0.8,0.1])
		for index,value in enumerate(dist):
			h.extend([index+1]*value)
		dist = multinomial(250,[0.8,0.2,0.1])
		for index,value in enumerate(dist):
			h.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f,g,h)))

	if n_columns == 7:
		dist = multinomial(250,[0.8,0.15,0.05])
		for index,value in enumerate(dist):
			i.extend([index+1]*value)
		dist = multinomial(250,[0.15,0.1,0.75])
		for index,value in enumerate(dist):
			i.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f,g,h,i)))

	if n_columns == 8:
		dist = multinomial(250,[0.55,0.4,0.05])
		for index,value in enumerate(dist):
			j.extend([index+1]*value)
		dist = multinomial(250,[0.05,0.8,0.15])
		for index,value in enumerate(dist):
			j.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f,g,h,i,j)))
		
	if n_columns == 9:
		dist = multinomial(250,[0.9,0.05,0.05])
		for index,value in enumerate(dist):
			k.extend([index+1]*value)
		dist = multinomial(250,[0.01,0.05,0.94])
		for index,value in enumerate(dist):
			k.extend([index+1]*value)
		data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b,c,d,e,f,g,h,i,j,k)))
	cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[0,0]+[1]*(n_columns))
	print(cluster_corr[2])
	print("Test dataset (with clusters - mixed values) - 500 samples. 2 continuous columns, {0} categorical column with 3 modalities".format(n_columns))
	diff = test_differences(*cluster_corr)
	local_out = diff + (n_columns,)
	out.append(local_out)

	del cluster_corr

print("----------")
print("Real datasets")
print("----------")

data_credit = pd.read_csv("./donnees/credit.csv",sep=",")
data_credit[['c1','c4','c5','c6','c7','c9','c10','c12','c13','c16']] = data_credit[['c1','c4','c5','c6','c7','c9','c10','c12','c13','c16']].apply(preprocessing.LabelEncoder().fit_transform).astype('float64')
data_credit.replace('?',np.nan, inplace=True)
data_credit.dropna(axis=0,how='any', inplace=True)
data_credit = data_credit.astype('float64')
credit_type = [1,0,0,1,1,1,1,0,1,1,0,1,1,0,0]
credit = (data_credit.iloc[:,:-1].values,data_credit.ix[:,-1].values, credit_type)
print("Credit dataset")
test_differences(*credit)
del credit

data_cmc = pd.read_csv("./donnees/cmc.csv",sep=",",header=None).astype('float64')
cmc_type = [0,1,1,0,1,1,1,1,1]
cmc = (data_cmc.iloc[:,:-1].values, data_cmc.iloc[:,-1].values,cmc_type)
print("CMC dataset - Num/Cat")
test_differences(*cmc)
del cmc


data_cmc = pd.read_csv("./donnees/cmc.csv",sep=",",header=None).astype('float64')
cmc_type = [0,0,0,0,1,1,1,0,1]
cmc = (data_cmc.iloc[:,:-1].values, data_cmc.iloc[:,-1].values,cmc_type)
print("CMC dataset - Num/Ord/Cat")
test_differences(*cmc)
del cmc
