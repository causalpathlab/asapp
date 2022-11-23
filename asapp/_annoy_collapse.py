from binascii import Incomplete
from glob import glob
from threading import local
import numpy as np
import pandas as pd
import annoy
import logging
logger = logging.getLogger(__name__)

class ApproxNN():
	def __init__(self, data, labels):
		self.dimension = data.shape[1]
		self.data = data.astype('float32')
		self.labels = labels

	def build(self, number_of_trees=50):
		self.index = annoy.AnnoyIndex(self.dimension,'angular')
		for i, vec in enumerate(self.data):
			self.index.add_item(i, vec.tolist())
		self.index.build(number_of_trees)

	def query(self, vector, k,wdist=False):
		indexes = self.index.get_nns_by_vector(vector.tolist(),k,include_distances=wdist)
		return indexes 
	
	def query_item(self, item, k,wdist=False):
		indexes = self.index.get_nns_by_item(item,k,include_distances=wdist)
		return indexes 

def get_ANNmodel(mtx,rows):
	model_ann = ApproxNN(mtx,rows)
	model_ann.build()
	return model_ann


'''
greedy algorithm-
We want to obtain entire hyperplane bounded region of a query point in the annoy forest as our collapsing region.

1. First, we get any random cell and query its neighbors. This set will be the first collapsed entry in a dictionary. We will mark all the point in this region as "occupied" points.

2. Shuffle and pick next point and its neighbors. 
    If all neighbors ( size equal to the user input neighbor size) are not "occupied" then mark this as new collapsing space in a dictionary.
	If any neighbor in this list are previously occupied then find the closet neighbor and assign this entire list to its respective space.
'''

def get_collapsed_neighbors_greedy(mtx,rows,nbrsize,sratio):
	m = get_ANNmodel(mtx,rows)
	
	collapsed_d = {}
	global_nbrs = []
	
	progress = 0 
	
	idx_list = np.arange(len(mtx))
	np.random.shuffle(idx_list)
	idx = idx_list[0]
	incomplete = True
	
	while incomplete:

		neighbors = np.array(m.query(mtx[idx],k=nbrsize))
		local_nbrs = []
		previous_nbr = None
		for n in neighbors: 
			# get all new neighbors
			if n not in global_nbrs:
				local_nbrs.append(n)
			# if found any previously assign neighbor then record this neighbor
			else:
				previous_nbr = n
		
		# if neighbors are all new then this is new entry
		if len(local_nbrs) == nbrsize:
			for ln in local_nbrs:global_nbrs.append(ln)
			collapsed_d[idx] = local_nbrs
			
			if len(global_nbrs) == len(rows):
				incomplete=False
		# if only subset is new neighbors then take furthest neighbor and assign local nbrs to its previous group
		else:
			for d in collapsed_d.keys():
				if previous_nbr in collapsed_d[d]:
					collapsed_d[d] = collapsed_d[d]+local_nbrs
					global_nbrs = global_nbrs + local_nbrs
				

				idx_list = [ x for x in idx_list if x not in local_nbrs]

		if len(idx_list)>0:
			np.random.shuffle(idx_list)
			idx = idx_list[0]
		else:
			incomplete=False
		
		if len(global_nbrs)>progress:
			print('collapsing...',len(global_nbrs))
			progress += 1000
			
		if len(global_nbrs) >= int(len(rows) * sratio):
			return collapsed_d,global_nbrs
		
	return collapsed_d,global_nbrs


def get_adjacency_neighbors(mtx,rows,nbrsize):
	m = get_ANNmodel(mtx,rows)
	nbrs = {}
	ndists = {}
	for idx in  np.arange(len(mtx)):
		# nbrs[idx],ndists[idx] = m.query(mtx[idx],k=nbrsize,wdist=True)
		nbrs[idx],ndists[idx] = m.query_item(idx,k=nbrsize,wdist=True)
	return nbrs,ndists



