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



def test_differences(data,Y,coltype,nmin=None):
	global_difference = []
	for i in range(20):
		matrices = []
		for j in range(5):
			similarity = build_ensemble_inc(data, coltypes=coltype,nmin=nmin)
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


out = []

# for n_columns in range(2,10):
#     local_out = ()
#     a = np.random.uniform(0,0.5,250)
#     a = np.append(a,np.random.uniform(0.5,1,250))
#     b = np.random.uniform(1,2,250)
#     b = np.append(b,np.random.uniform(0,1,250))
#     for i in range(n_columns):
#         dist = multinomial(250,[0.8])
#         c = []
#         for index,value in enumerate(dist):
#             c.extend([index+1]*value)
#         dist = multinomial(250,[0.2])
#         for index,value in enumerate(dist):
#             c.extend([index+1]*value)

#     data_cluster_corr = pd.DataFrame(data = {'a':a,'b':b,'c':c})
#     cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[0,0]+[1]*(n_columns-2))
#     print("Test dataset (with clusters - mixed values) - (500, 3). Categorical column with {0} modalities".format(n_columns))
#     diff = test_differences(*cluster_corr)
#     local_out = diff + (n_columns,)
#     out.append(local_out)

#     del cluster_corr

# print(out)
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# x_ax = [value[2] for value in out]

# noise_mean = [value[0] for value in out]
# noise_std = [value[1] for value in out]
# plt.ylim(0,1)

# plt.errorbar(x_ax,noise_mean,yerr = noise_std)

# plt.xlabel("Number of modalities")
# plt.ylabel("Mean difference between intracluster and intercluster similarities")
# plt.savefig("n_modalities.png")
# plt.clf()
# plt.cla()
# plt.close()

import matplotlib

for nmin in range(3,5):

    a = np.random.uniform(0,0.5,250)
    a = np.append(a,np.random.uniform(0.5,1,250))
    b = np.random.uniform(1,2,250)
    b = np.append(b,np.random.uniform(0,1,250))
    c = []
    d = []
    e = []
    f = []
    g = []
    h = []
    i = []
    j = []
    k = []
    out = []
    data_cluster_corr = pd.DataFrame(data = np.column_stack((a,b)))
    for n_columns in range(0,9):
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
        cluster_corr = (data_cluster_corr.values.astype('float64'),[0 if i < 250 else 1 for i in range(len(data_cluster_corr))],[0,0]+[1]*(n_columns),len(data_cluster_corr.values.astype('float64'))/nmin)
        print(cluster_corr[2])
        print("Test dataset (with clusters - mixed values) - (500, 3). {0} categorical column with 3 modalities".format(n_columns))
        diff = test_differences(*cluster_corr)
        local_out = diff + (n_columns,)
        out.append(local_out)

        del cluster_corr

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    x_ax = [value[2] for value in out]

    noise_mean = [value[0] for value in out]
    noise_std = [value[1] for value in out]
    plt.ylim(0,1)

    plt.errorbar(x_ax,noise_mean,yerr = noise_std,label=nmin)

plt.xlabel("Number of columns")
plt.ylabel("Mean difference between intracluster and intercluster similarities")
plt.legend()
plt.savefig("n_columns.png")
plt.clf()
plt.cla()
plt.close()