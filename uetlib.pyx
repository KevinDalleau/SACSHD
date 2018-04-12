# cython: profile=True
cimport numpy as np2
import numpy as np
import math
import random
from libc.math cimport isnan
import cython
from cython.parallel import parallel, prange
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)

cdef np2.ndarray[np2.int16_t, ndim=2] compute_similarity(np2.ndarray[np2.float64_t, ndim=2] data, leaves):
	matrix = [[0 for i in range(len(data))] for j in range(len(data))]
	cdef np2.int16_t numLeaves = len(leaves)
	cdef np2.int16_t numLocalLeaves

	for l1 in range(numLeaves):
		leaf = leaves[l1]
		numLocalLeaves = len(leaf)
		for i1 in range(numLocalLeaves):
			p1 = leaf[i1]
			for i2 in range(numLocalLeaves):
				p2 = leaf[i2]
				matrix[p1][p2]+=1
	return(np.array(matrix))

cdef np2.ndarray[np2.int16_t, ndim=2] compute_similarity_inc(np2.int32_t size, leaves):
	matrix = [[0 for i in range(size)] for j in range(size)]
	cdef np2.int16_t numLocalLeaves

	while leaves:
		leaf = leaves.pop()
		numLocalLeaves = len(leaf)
		for i1 in range(numLocalLeaves):
			p1 = leaf[i1]
			for i2 in range(numLocalLeaves):
				p2 = leaf[i2]
				matrix[p1][p2]+=1
	return(np.array(matrix))


cdef np2.int16_t get_type(column,n):
	cdef np2.int16_t type = 2
	unique_values = len(set(np.array(column)[~np.isnan(column)]))
	if(unique_values > n):
		type = 0 # "num"
	else:
		type = 1 # "nom"
	return(type)

cdef float random_split(values,type):
	if type==0:
		split = np.random.uniform(np.mean(values),np.std(values))
	if type==1:
		split = random.choice(values)
	return(split)


cdef build_randomized_tree2_cont_cat(data, nmin, coltypes):

	cdef np2.ndarray[np2.int16_t, ndim=1] instanceslist = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] indices = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes_indices = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)

	#cdef np2.ndarray[np2.int16_t, ndim=1] coltypes = np.zeros(len(attributes),dtype=np.int16)

	#for i in attributes:
	#	coltypes[i]= get_type(data[:,attributes[i]],n)

	if not nmin:
		nmin = math.floor(len(data)/3)

	leaves = []
	nodes = []
	cdef np2.ndarray[np2.long_t, ndim=1] continuous_indices = np.where(np.array(coltypes) == 0)[0]
	# print(continuous_indices)
	cdef np2.int_t rand_index
	cdef np2.int_t attribute
	cdef np2.int_t col_type
	
	if len(continuous_indices)!=0:
		# print("Two types of attributes. Beginning by continuous ones.\n")
		rand_index = random.choice(continuous_indices)
		continuous_indices = np.delete(continuous_indices,np.argwhere(continuous_indices==rand_index),0)
		attribute = attributes[rand_index]
		# print(rand_index)
		
	else:
		rand_index = random.choice(attributes_indices)
		attribute = attributes[rand_index]
	
	cdef np2.ndarray[np2.float64_t, ndim=1] col = data[:,attributes[rand_index]]
	attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)

	col_type = coltypes[rand_index]


	split_value = random_split(col,col_type)

	if col_type == 0:
		node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
	else:
		node_left_indices = np.where(col == split_value)[0]
	node_left_instances = [instanceslist[i] for i in node_left_indices] # Instances in left node
	node_left_d = data[node_left_indices,:] # Data in left node

	node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices],dtype=np.int16) # Indices of instances in right node
	node_right_instances = [instanceslist[i] for i in node_right_indices] # Instances in right node
	node_right_d = data[node_right_indices,:] # Data in right node
	if(len(node_left_indices) < nmin):
		leaves.append(node_left_instances)
	else:
		nodes.append((node_left_indices,node_left_instances,node_left_d))
	if(len(node_right_indices) < nmin):
		leaves.append(node_right_instances)
	else:
		nodes.append((node_right_indices,node_right_instances,node_right_d))

	while(len(nodes)!=0):

		if(len(attributes_indices) < 1):
			for i in range(len(nodes)):
				node_instances = nodes[i][1]
				leaves.append(node_instances)
			break
		node_indices, node_instances, node_data = nodes.pop(0)



		if len(continuous_indices)!=0:
			# print("Two types of attributes. Beginning by continuous ones.\n")
			rand_index = random.choice(continuous_indices)
			continuous_indices = np.delete(continuous_indices,np.argwhere(continuous_indices==rand_index),0)
			
		else:
			# print("End of continuous attributes")
			rand_index = random.choice(attributes_indices)
	
		rand_index = random.choice(attributes_indices)

		attribute = attributes[rand_index]

		col = node_data[:,attributes[rand_index]]
		attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)
		col_type = coltypes[rand_index]
		col = np.array(col)

		# print(col)
		if(len(node_indices) >= nmin and len(set(col))>1):
			split_value = random_split(col,col_type)
			# print("Column type : {0}".format(col_type))
			if col_type == 0:
				node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
			else:
				node_left_indices = np.where(col == split_value)[0]
			# print("Node left instances : {0}".format(node_left_instances))
			# print("Node left indices : {0}".format(node_left_indices))

			node_left_instances = [node_instances[i] for i in node_left_indices] # Instances in left node
			node_left_d = data[node_left_indices,:] # Data in left node

			node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices], dtype=np.int16) # Indices of instances in right node

			node_right_instances = [node_instances[i] for i in node_right_indices] # Instances in right node
			node_right_d = data[node_right_indices,:] # Data in right node

			if(len(node_left_indices) < nmin):
				leaves.append(node_left_instances)
			else:
				nodes.append((node_left_indices,node_left_instances,node_left_d))
			if(len(node_right_indices) < nmin):
				leaves.append(node_right_instances)
			else:
				nodes.append((node_right_indices,node_right_instances,node_right_d))
		else:
			leaves.append(node_instances)
	return(np.array(leaves, dtype = np.ndarray))


cdef build_randomized_tree2_cat_cont(data, nmin, coltypes):

	cdef np2.ndarray[np2.int16_t, ndim=1] instanceslist = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] indices = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes_indices = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)

	#cdef np2.ndarray[np2.int16_t, ndim=1] coltypes = np.zeros(len(attributes),dtype=np.int16)

	#for i in attributes:
	#	coltypes[i]= get_type(data[:,attributes[i]],n)

	if not nmin:
		nmin = math.floor(len(data)/3)

	leaves = []
	nodes = []
	cdef np2.ndarray[np2.long_t, ndim=1] categorical_indices = np.where(np.array(coltypes) == 1)[0]
	cdef np2.int_t rand_index
	cdef np2.int_t attribute
	cdef np2.int_t col_type
	
	if len(categorical_indices)!=0:
		rand_index = random.choice(categorical_indices)
		categorical_indices = np.delete(categorical_indices,np.argwhere(categorical_indices==rand_index),0)
		attribute = attributes[rand_index]
		# print(rand_index)
		
	else:
		rand_index = random.choice(attributes_indices)
		attribute = attributes[rand_index]
	
	cdef np2.ndarray[np2.float64_t, ndim=1] col = data[:,attributes[rand_index]]
	attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)

	col_type = coltypes[rand_index]


	split_value = random_split(col,col_type)

	if col_type == 0:
		node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
	else:
		node_left_indices = np.where(col == split_value)[0]
	node_left_instances = [instanceslist[i] for i in node_left_indices] # Instances in left node
	node_left_d = data[node_left_indices,:] # Data in left node

	node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices],dtype=np.int16) # Indices of instances in right node
	node_right_instances = [instanceslist[i] for i in node_right_indices] # Instances in right node
	node_right_d = data[node_right_indices,:] # Data in right node
	if(len(node_left_indices) < nmin):
		leaves.append(node_left_instances)
	else:
		nodes.append((node_left_indices,node_left_instances,node_left_d))
	if(len(node_right_indices) < nmin):
		leaves.append(node_right_instances)
	else:
		nodes.append((node_right_indices,node_right_instances,node_right_d))

	while(len(nodes)!=0):

		if(len(attributes_indices) < 1):
			for i in range(len(nodes)):
				node_instances = nodes[i][1]
				leaves.append(node_instances)
			break
		node_indices, node_instances, node_data = nodes.pop(0)



		if len(categorical_indices)!=0:
			# print("Two types of attributes. Beginning by continuous ones.\n")
			rand_index = random.choice(categorical_indices)
			categorical_indices = np.delete(categorical_indices,np.argwhere(categorical_indices==rand_index),0)
			
		else:
			# print("End of continuous attributes")
			rand_index = random.choice(attributes_indices)
	
		rand_index = random.choice(attributes_indices)

		attribute = attributes[rand_index]

		col = node_data[:,attributes[rand_index]]
		attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)
		col_type = coltypes[rand_index]
		col = np.array(col)

		# print(col)
		if(len(node_indices) >= nmin and len(set(col))>1):
			split_value = random_split(col,col_type)
			# print("Column type : {0}".format(col_type))
			if col_type == 0:
				node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
			else:
				node_left_indices = np.where(col == split_value)[0]
			# print("Node left instances : {0}".format(node_left_instances))
			# print("Node left indices : {0}".format(node_left_indices))

			node_left_instances = [node_instances[i] for i in node_left_indices] # Instances in left node
			node_left_d = data[node_left_indices,:] # Data in left node

			node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices], dtype=np.int16) # Indices of instances in right node

			node_right_instances = [node_instances[i] for i in node_right_indices] # Instances in right node
			node_right_d = data[node_right_indices,:] # Data in right node

			if(len(node_left_indices) < nmin):
				leaves.append(node_left_instances)
			else:
				nodes.append((node_left_indices,node_left_instances,node_left_d))
			if(len(node_right_indices) < nmin):
				leaves.append(node_right_instances)
			else:
				nodes.append((node_right_indices,node_right_instances,node_right_d))
		else:
			leaves.append(node_instances)
	return(np.array(leaves, dtype = np.ndarray))

cdef build_randomized_tree2(data, nmin, coltypes):

	cdef np2.ndarray[np2.int16_t, ndim=1] instanceslist = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] indices = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes_indices = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)
	# print(attributes_indices)
	#cdef np2.ndarray[np2.int16_t, ndim=1] coltypes = np.zeros(len(attributes),dtype=np.int16)

	#for i in attributes:
	#	coltypes[i]= get_type(data[:,attributes[i]],n)

	if not nmin:
		nmin = math.floor(len(data)/3)

	leaves = []
	nodes = []
	cdef np2.int_t rand_index = random.choice(attributes_indices)
	cdef np2.int_t attribute = attributes[rand_index]

	cdef np2.ndarray[np2.float64_t, ndim=1] col = data[:,attributes[rand_index]]
	attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)

	cdef np2.int_t col_type = coltypes[rand_index]

	split_value = random_split(col,col_type)

	if col_type == 0:
		node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
	else:
		node_left_indices = np.where(col == split_value)[0]
	node_left_instances = [instanceslist[i] for i in node_left_indices] # Instances in left node
	node_left_d = data[node_left_indices,:] # Data in left node

	node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices],dtype=np.int16) # Indices of instances in right node
	node_right_instances = [instanceslist[i] for i in node_right_indices] # Instances in right node
	node_right_d = data[node_right_indices,:] # Data in right node
	if(len(node_left_indices) < nmin):
		leaves.append(node_left_instances)
	else:
		nodes.append((node_left_indices,node_left_instances,node_left_d))
	if(len(node_right_indices) < nmin):
		leaves.append(node_right_instances)
	else:
		nodes.append((node_right_indices,node_right_instances,node_right_d))

	while(len(nodes)!=0):

		if(len(attributes_indices) < 1):
			for i in range(len(nodes)):
				node_instances = nodes[i][1]
				leaves.append(node_instances)
			break
		node_indices, node_instances, node_data = nodes.pop(0)
		rand_index = random.choice(attributes_indices)

		attribute = attributes[rand_index]

		col = node_data[:,attributes[rand_index]]
		attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)
		col_type = coltypes[rand_index]
		col = np.array(col)
		if(len(node_indices) >= nmin and len(set(col))>1):
			split_value = random_split(col,col_type)

			if col_type == 0:
				node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
			else:
				node_left_indices = np.where(col == split_value)[0]

			node_left_instances = [node_instances[i] for i in node_left_indices] # Instances in left node
			node_left_d = data[node_left_indices,:] # Data in left node

			node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices], dtype=np.int16) # Indices of instances in right node

			node_right_instances = [node_instances[i] for i in node_right_indices] # Instances in right node
			node_right_d = data[node_right_indices,:] # Data in right node

			if(len(node_left_indices) < nmin):
				leaves.append(node_left_instances)
			else:
				nodes.append((node_left_indices,node_left_instances,node_left_d))
			if(len(node_right_indices) < nmin):
				leaves.append(node_right_instances)
			else:
				nodes.append((node_right_indices,node_right_instances,node_right_d))
		else:
			leaves.append(node_instances)
	return(np.array(leaves, dtype = np.ndarray))









from scipy.sparse import dok_matrix

cdef np2.ndarray[np2.float_t, ndim=2] build_randomized_tree_and_get_sim(data, nmin, coltypes):
	# matrix = np.array([[0 for i in range(len(data))] for j in range(len(data))])
	matrix = np.zeros((len(data),len(data)))
	# matrix = dok_matrix((len(data),len(data)))
	cdef np2.ndarray[np2.int16_t, ndim=1] instanceslist = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] indices = np.linspace(0,len(data)-1,len(data),dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)
	cdef np2.ndarray[np2.int16_t, ndim=1] attributes_indices = np.linspace(0,data.shape[1]-1,data.shape[1],dtype=np.int16)
	# print(attributes_indices)
	#cdef np2.ndarray[np2.int16_t, ndim=1] coltypes = np.zeros(len(attributes),dtype=np.int16)

	#for i in attributes:
	#	coltypes[i]= get_type(data[:,attributes[i]],n)

	if not nmin:
		nmin = math.floor(len(data)/3)

	leaves = []
	nodes = []
	cdef np2.int_t rand_index = random.choice(attributes_indices)
	cdef np2.int_t attribute = attributes[rand_index]

	cdef np2.ndarray[np2.float64_t, ndim=1] col = data[:,attributes[rand_index]]
	attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)

	cdef int col_type = coltypes[rand_index]

	cdef float split_value = random_split(col,col_type)

	if col_type == 0:
		node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
	else:
		node_left_indices = np.where(col == split_value)[0]
	node_left_instances = [instanceslist[i] for i in node_left_indices] # Instances in left node
	node_left_d = data[node_left_indices,:] # Data in left node

	node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices],dtype=np.int16) # Indices of instances in right node
	node_right_instances = [instanceslist[i] for i in node_right_indices] # Instances in right node
	node_right_d = data[node_right_indices,:] # Data in right node
	if(len(node_left_indices) < nmin):
		numLocalLeaves = len(node_left_instances)
		for i1 in range(numLocalLeaves):
			p1 = node_left_instances[i1]
			for i2 in range(numLocalLeaves):
				p2 = node_left_instances[i2]
				matrix[p1][p2]+=1
	else:
		nodes.append((node_left_indices,node_left_instances,node_left_d))
	if(len(node_right_indices) < nmin): # We are in a leaf node
		numLocalLeaves = len(node_right_instances)
		for i1 in range(numLocalLeaves):
			p1 = node_right_instances[i1]
			for i2 in range(numLocalLeaves):
				p2 = node_right_instances[i2]
				matrix[p1][p2]+=1
	else:
		nodes.append((node_right_indices,node_right_instances,node_right_d))

	while(len(nodes)!=0):

		if(len(attributes_indices) < 1):
			for i in range(len(nodes)):
				node_instances = nodes[i][1]
				numLocalLeaves = len(node_instances)
				for i1 in range(numLocalLeaves):
					p1 = node_instances[i1]
					for i2 in range(numLocalLeaves):
						p2 = node_instances[i2]
						matrix[p1][p2]+=1
			break
		node_indices, node_instances, node_data = nodes.pop(0)
		rand_index = random.choice(attributes_indices)

		attribute = attributes[rand_index]

		col = node_data[:,attributes[rand_index]]
		attributes_indices = np.delete(attributes_indices,np.argwhere(attributes_indices==rand_index),0)
		col_type = coltypes[rand_index]
		col = np.array(col)
		if(len(node_indices) >= nmin and len(set(col))>1):
			split_value = random_split(col,col_type)

			if col_type == 0:
				node_left_indices = np.where(col < split_value)[0] # Indices of instances in left
			else:
				node_left_indices = np.where(col == split_value)[0]

			node_left_instances = [node_instances[i] for i in node_left_indices] # Instances in left node
			node_left_d = data[node_left_indices,:] # Data in left node

			node_right_indices = np.array([i for i in range(len(col)) if i not in node_left_indices], dtype=np.int16) # Indices of instances in right node

			node_right_instances = [node_instances[i] for i in node_right_indices] # Instances in right node
			node_right_d = data[node_right_indices,:] # Data in right node

			if(len(node_left_indices) < nmin):
				numLocalLeaves = len(node_left_instances)
				for i1 in range(numLocalLeaves):
					p1 = node_left_instances[i1]
					for i2 in range(numLocalLeaves):
						p2 = node_left_instances[i2]
						matrix[p1][p2]+=1
			else:
				nodes.append((node_left_indices,node_left_instances,node_left_d))
			if(len(node_right_indices) < nmin):
				numLocalLeaves = len(node_right_instances)
				for i1 in range(numLocalLeaves):
					p1 = node_right_instances[i1]
					for i2 in range(numLocalLeaves):
						p2 = node_right_instances[i2]
						matrix[p1][p2]+=1
			else:
				nodes.append((node_right_indices,node_right_instances,node_right_d))
		else:
			numLocalLeaves = len(node_instances)
			for i1 in range(numLocalLeaves):
				p1 = node_instances[i1]
				for i2 in range(numLocalLeaves):
					p2 = node_instances[i2]
					matrix[p1][p2]+=1
	return(matrix)

def get_sim(data,nmin,coltypes):
	leaves = build_randomized_tree2(data,nmin,coltypes)
	return(compute_similarity(data,leaves))

def get_sim_inc(data,nmin,coltypes):
	leaves = build_randomized_tree2(data,nmin,coltypes)
	return(compute_similarity_inc(len(data),leaves))

def get_sim_one(data,nmin,coltypes):
	return(build_randomized_tree_and_get_sim(data,nmin,coltypes))

def get_sim_cont_cat(data,nmin,coltypes):
	leaves = build_randomized_tree2_cont_cat(data, nmin,coltypes)
	return(compute_similarity(data,leaves))

def get_sim_cat_cont(data,nmin,coltypes):
	leaves = build_randomized_tree2_cat_cont(data, nmin,coltypes)
	return(compute_similarity(data,leaves))
