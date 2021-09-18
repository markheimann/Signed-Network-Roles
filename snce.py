
'''
Code adapted from implementations of REGAL [1] and EMBER [2].
[1] Mark Heimann, Haoming Shen, Tara Safavi, and Danai Koutra. 
REGAL: Representation Learning-based Graph Alignment. CIKM 2018.
[2] Di Jin*, Mark Heimann*, Tara Safavi, Mengdi Wang, Wei Lee, Lindsay Snider, Danai Koutra. 
Smart Roles: Inferring Professional Roles in Email Networks. KDD 2019.
'''

import numpy as np 
from scipy import sparse
import pandas as pd
import networkx as nx
import os, sys, time, math
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict

#Get (signed, outgoing) neighborhoods of each node up to max_hop steps
def get_neighborhoods(signed_adj_list, max_hop = 2):
	neighborhoods = {}

	#Initialize 0-hop neighborhood has itself as a positive connection
	for i in signed_adj_list:
		neighborhoods[i] = {"positive":{0:[i]}, "negative":{0:[]}}

		#Higher order neighborhoods
		for l in range(1, max_hop + 1):
			#Friends of friends or enemies of enemies (positive balance)
			friends_of_friends = [set(signed_adj_list[pos_neigh]["positive"]) for pos_neigh in neighborhoods[i]["positive"][l-1] ] + [set()]
			enemies_of_enemies = [set(signed_adj_list[neg_neigh]["negative"]) for neg_neigh in neighborhoods[i]["negative"][l-1] ] + [set()]

			#Friends of enemies or enemies of friends (negative balance)
			enemies_of_friends = [set(signed_adj_list[pos_neigh]["negative"]) for pos_neigh in neighborhoods[i]["positive"][l-1] ] + [set()]
			friends_of_enemies = [set(signed_adj_list[neg_neigh]["positive"]) for neg_neigh in neighborhoods[i]["negative"][l-1] ] + [set()]

			neighborhoods[i]["positive"][l] = set.union(*friends_of_friends) \
												.union(*enemies_of_enemies)

			neighborhoods[i]["negative"][l] = set.union(*enemies_of_friends) \
												.union(*friends_of_enemies)

	return neighborhoods

#Input: adj matrix
#Output: list of positive and negative neighbors for each node
#Format: dict { int : {"positive": np arr of ints}, {"negative": np arr of ints}}
def compute_signed_adj_list(signed_adj):
	signed_adj = signed_adj.tocoo()
	adj_list = {}
	for i in range(signed_adj.shape[0]):
		adj_list[i] = {"positive":[], "negative":[]}
	df = pd.DataFrame(data = {"row":signed_adj.row, "col":signed_adj.col, "data":signed_adj.data})
	grouping = df.groupby(["data", "row"]).groups
	for group in grouping:
		if group[0] < 0:
			sign = "negative"
		else:
			sign = "positive"
		adj_list[group[1]][sign] = df.loc[grouping[group] ]["col"].tolist()
	return adj_list

#Write down degrees for neighbors
#base 1: no logarithmic binning
def get_sequences(neighborhood, feature_vals, max_feat, base = 2):
	if base is None: base = 1

	feat_hist = np.asarray([0] * int(math.log(max(max_feat, 1), base) + 1))
	for kn in neighborhood:
		try:
			feat_hist[int(math.log(feature_vals[kn], base))] += 1 #Note: could be some other weight e.g. pathweight (EMBER)
		except:
			pass
			#print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
	return feat_hist

#Discount 0.5 for synthetic to make neighborhood distinctions clearer
#Discount 0.1 for real? as in the past
def get_features(neighborhoods, signed_adj_list, base = 2, discount = 0.1):
	num_nodes = len(neighborhoods)

	pos_outdegree = np.asarray([len(signed_adj_list[kn]["positive"]) for kn in neighborhoods])
	neg_outdegree = np.asarray([len(signed_adj_list[kn]["negative"]) for kn in neighborhoods])
	max_posout = np.max(pos_outdegree)
	max_negout = np.max(neg_outdegree)

	base_features = dict()
	base_features["pos_out"] = {"vals":pos_outdegree, "max":max_posout}
	base_features["neg_out"] = {"vals":neg_outdegree, "max":max_negout}

	features = {}
	for base_feat in base_features.keys():
		try:
			num_feat = int(math.log(max(base_features[base_feat]["max"], 1), base) + 1)
			features[base_feat] = {"positive": np.zeros((num_nodes, num_feat)), "negative":np.zeros((num_nodes, num_feat))}
		except Exception as e:
			print(base_features[base_feat]["max"])
			raise ValueError(e)
	for n in range(num_nodes): #for each node
		for sign in ["positive", "negative"]:
			for l in range(len(neighborhoods[n][sign])):
				for base_feat in base_features.keys():
					if len(neighborhoods[n][sign][l]) > 0:
						#degree sequence of node n at layer "layer"
						deg_seq = get_sequences(neighborhoods[n][sign][l], base_features[base_feat]["vals"], base_features[base_feat]["max"], base = base)
						#add degree info from this degree sequence, weighted depending on layer and discount factor alpha
						features[base_feat][sign][n] += (discount**l) * deg_seq

	features = np.hstack([features[base_feat][sign] for base_feat in base_features.keys() for sign in ["positive", "negative"]]) #Combine positive and negative features separately
	return features

#features: node's local structural info
#Compare to d landmarks to derive embeddings
def sim_to_embed(features, landmark_indices = None, d = 128, normalize = False):
	n = features.shape[0]
	if landmark_indices is not None: d = len(landmark_indices)
	if landmark_indices is None: np.random.seed(42); landmark_indices = np.random.permutation(np.arange(n))[:d] #select d random landmarks

	C = euclidean_distances(features, features[landmark_indices])
	C = np.exp(-C)

	#Compute Nystrom-based node embeddings
	W_pinv = np.linalg.pinv(C[landmark_indices])
	U,X,V = np.linalg.svd(W_pinv)
	Wfac = np.dot(U, np.diag(np.sqrt(X)))
	del U,X,V
	reprsn = np.dot(C, Wfac)

	#Post-processing step to normalize embeddings (true by default)
	if normalize:
		reprsn = reprsn / np.linalg.norm(reprsn, axis = 1).reshape((reprsn.shape[0],1))
	return reprsn

#Full embedding pipeline
def learn_embeddings(signed_adj, directed = True, dim = 8):
	#For directed repeated these step on transpose and concatenate features (as in EMBER)
	#========
	if directed:
		directions = ["out", "in"]
	else:
		directions = ["out"]
	features = []

	for direction in directions:
		print("%sdegree structural identity computation..." % direction)
		if direction == "in":
			S = signed_adj.T
		else:
			S = signed_adj

		before_preproc = time.time()
		signed_adj_list = compute_signed_adj_list(S) #preprocess by computing signed adjacency list
		print("Preprocessed signed adjacency lists in time:", time.time() - before_preproc)

		before_neighborhoods = time.time()
		neighborhoods = get_neighborhoods(signed_adj_list)
		print("Extracted signed neighborhoods in time:", time.time() - before_neighborhoods)

		before_feat = time.time()
		features.append(get_features(neighborhoods, signed_adj_list))
		print("Signed structural identity in time:", time.time() - before_feat)

	features = np.hstack(features)

	before_emb = time.time()
	reprsn = sim_to_embed(features, d = dim)
	print("Learned embedding in time:", time.time() - before_emb)

	return reprsn

if __name__ == "__main__":
	#Small, undirected
	adj = nx.adjacency_matrix( nx.read_edgelist("data/ucidata-gama.edgelist", comments="%", create_using = nx.Graph(), data=(('weight',float),)) ) #signed undirected network
	emb = learn_embeddings(adj)
	print(emb.shape)

	#Large, directed
	adj = nx.adjacency_matrix( nx.read_edgelist("data/slashdot-zoo.edgelist", comments="%", create_using = nx.DiGraph(), data=(('weight',float),)) )#signed directed network
	emb = learn_embeddings(adj)
	print(emb.shape)

