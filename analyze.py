import numpy as np 
import networkx as nx 
from scipy import sparse

import os, sys, time

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

import snce, srde
from utils import *

#Embedding pipeline 
def embed_data(graph, method = "snce", dim = 128):
	if method == "snce":
		return snce.learn_embeddings(graph, dim = dim)
	elif method == "srde":
		return srde.learn_embeddings(graph, dim = dim)

	#NOTE code for running baselines
	INPUT_FILE = "graph.edgelist"
	OUTPUT_FILE = "emb.emb"

	if method == "xnetmf":
		emb_dir = "../empirical_replearning" 
		cmd = "python main.py --input %s --output %s --method %s --dimension %d" % (INPUT_FILE, OUTPUT_FILE, method, dim)
		delim = " "
	elif method == "sgcn":
		emb_dir = "../SGCN-master"
		cmd = "python src/main.py --reduction-dimensions %d --edge-path %s --features-path %s --embedding-path %s" % (dim, INPUT_FILE, INPUT_FILE, OUTPUT_FILE)
		delim = ","
	else:
		raise NotImplementedError

	cwd = os.getcwd()
	os.chdir(emb_dir) 

	#Write graph to edgelist
	nx.write_edgelist(to_nx(graph), INPUT_FILE)

	before_emb = time.time()
	#embed graph
	print(cmd)
	os.system(cmd)
	print("embedded network in time %.2f" % (time.time() - before_emb))
	#Read in embeddings
	emb = read_in_node2vec_format(OUTPUT_FILE, delim)

	#Clean up
	for fname in [INPUT_FILE, OUTPUT_FILE]:
		os.system("rm %s" % fname)

	os.chdir(cwd)

	return emb

#Baseline methods (hand-engineered features, single-sign embedding methods with or without sec- framework)
def signed_network_features(signed_adj, method = "sec-xnetmf", dim = 128):
	if method == "degrees":
		dim = 4
		#Construct positive and negative network
		pos_network, neg_network = split_signed_network(signed_adj)
		#Construct degree statistics
		all_posoutdeg = pos_network.sum(axis = 1)
		all_negoutdeg = neg_network.sum(axis = 1)
		all_posindeg = pos_network.sum(axis = 0)
		all_negindeg = neg_network.sum(axis = 0)

		#Aggregate as features
		features = np.zeros((signed_adj.shape[0], 4))
		features[:,0] = np.ravel(all_posoutdeg)
		features[:,1] = np.ravel(all_negoutdeg)
		features[:,2] = np.ravel(all_posindeg)
		features[:,3] = np.ravel(all_negindeg)

	elif method.startswith("xnetmf") or method.startswith("sec"):
		pos_network, neg_network = split_signed_network(signed_adj)
		if method.endswith("neg"):
			features = embed_data(neg_network, dim = dim, method = "xnetmf", N = signed_adj.shape[0])
		elif method.endswith("pos"):
			features = embed_data(pos_network, dim = dim, method = "xnetmf", N = signed_adj.shape[0])
		elif method.startswith("sec"):
			emb_neg = emb2mat(embed_data(neg_network, dim = int(dim/2), method = "xnetmf"), N = signed_adj.shape[0]) #assign the first dim/2 features to the positive embedding portion
			emb_pos = emb2mat(embed_data(pos_network, dim = int(dim - dim/2), method = "xnetmf"), N = signed_adj.shape[0]) #assign the rest to be the negative embedding portion
			print(neg_network.shape, pos_network.shape, emb_neg.shape, emb_pos.shape)
			features = np.hstack((emb_neg, emb_pos))

	elif method.startswith("sgcn"):
		features = emb2mat(embed_data(neg_network, method = "sgcn", dim = dim))

	return features

#Visualization
def viz_embed(features, method = "snce", viz_list = None, colors = None, show = False, subsample = True, dataset = "slashdot-zoo", viz_method = "PCA"):
	plt.cla()
	synth = (dataset.startswith("synthetic"))

	viz_2nodes = (len(viz_list) == 2)

	if viz_2nodes: #for the experiment where we want to tell apart just 2 nodes
		COLOR_HIGHLIGHT = "#d7191c" #med red
		COLOR_HIGHLIGHT2 = "#fdae61" #salmon
		COLOR_REG = "#2c7bb6" #med blue
		COLOR_REG2 = "#abd9e9" #light blue
	else:
		COLOR_HIGHLIGHT = "red" #nodes of interest, e.g. trolls
		COLOR_REG = "green" #the rest of the nodes in the graph

	#Get troll features and subsample nontroll features
	if colors is None and viz_list is not None:
		print("troll visualization...")
		#Get equal number of "regular" nodes
		regular_node_ids = np.setdiff1d(np.arange(features.shape[0]), viz_list)
		np.random.seed(0) #fix from run to run
		if subsample:
			regular_data = np.random.permutation(regular_node_ids)[:len(viz_list)]
		else:
			regular_data = regular_node_ids

		#Get features for nodes of interest and regular nodes
		viz_node_features = features[viz_list]
		regular_node_features = features[regular_data]

		#Reduce features to only nodes of interest and subsampled regular nodes
		features = np.vstack((viz_node_features, regular_node_features))

		#Color code by nodes of interest and not nodes of interest
		if viz_2nodes:
			colors = [COLOR_HIGHLIGHT, COLOR_HIGHLIGHT2] + [COLOR_REG] * len(regular_data)
		else:
			colors = [COLOR_HIGHLIGHT]*len(viz_list) + [COLOR_REG] * len(regular_data)
			
	if viz_method == "PCA":
		viz = PCA(n_components = 2, random_state = 42)
	else:
		viz_method = "t-SNE"
		viz = TSNE(random_state = 42)
	print("Using %s to learn 2D embedding for visualization..." % viz_method)
	before_dim_reduction = time.time()
	viz_2d = viz.fit_transform(features)
	print("Ran %s for  visualization in time" % viz_method, time.time() - before_dim_reduction)

	point_size = 250
	for i in range(len(viz_2d)):
		plt.scatter(viz_2d[i,0], viz_2d[i,1], color = colors[i], s = point_size)
		if not synth: #plot labels of nodes
			plt.text(viz_2d[i,0]+0.01*np.random.randint(10), viz_2d[i,1]+0.01*np.random.randint(10), i, fontsize = 24)
	
	plt.xticks([])
	plt.yticks([])

	if show:
		plt.show()
	else:
		plt.savefig("figs/%s/%s_features.png" % (dataset, method))

#Node classification
def classify_troll(features, viz_list):
	nontroll_ids = np.setdiff1d(np.arange(features.shape[0]), viz_list)
	np.random.seed(0) #fix from run to run
	nontroll_data = np.random.permutation(nontroll_ids)[:len(viz_list)]

	#Concatenate features/labels for trolls and nontroll data
	troll_features = features[viz_list]
	nontroll_features = features[nontroll_data]
	all_features = np.vstack((troll_features, nontroll_features))
	labels = np.append( np.zeros(len(viz_list)), np.ones(len(viz_list)) )

	n_fold = 10
	cv_score = cross_val_score(LogisticRegression(), all_features, labels, cv=n_fold, scoring = "accuracy")
	print("Mean accuracy across %d folds of CV: %.2f" % (n_fold, np.mean(cv_score)))

if __name__ == "__main__":
	nx_graph, viz_list = read_slashdot_data()
	snce_emb = embed_data(nx.adjacency_matrix(nx_graph), method = "snce")
	classify_troll(snce_emb, viz_list)