import numpy as np
from scipy import sparse
import networkx as nx

#========UTILS FOR GRAPH MANIPULATION===========
def create_combined_graph(graphs):
	dim_starts = [0] #where to start new graph
	for g in graphs:
		dim_starts.append(g.shape[0] + dim_starts[-1])

	combined_row = np.asarray([])
	combined_col = np.asarray([])
	combined_data = np.asarray([])

	for i in range(len(graphs)):	
		G_adj = graphs[i].tocoo()
		combined_row = np.concatenate((combined_row, G_adj.row + dim_starts[i]))
		combined_col = np.concatenate((combined_col, G_adj.col + dim_starts[i]))
		combined_data = np.concatenate((combined_data, G_adj.data))

	combined_shape = (dim_starts[-1], dim_starts[-1])
	combined_adj = sparse.coo_matrix((combined_data, (combined_row, combined_col)), shape = combined_shape).tocsr()

	return combined_adj, dim_starts

#Wrapper to create NetworkX graph from sparse or dense matrix
def to_nx(adjmat, directed = False):
	graph_type = None #will default to undirected
	if directed:
		graph_type = nx.DiGraph()
	if sparse.issparse(adjmat):
		return nx.from_scipy_sparse_matrix(adjmat, create_using = graph_type)
	else:
		return nx.from_numpy_matrix(adjmat, create_using = graph_type)

def to_undirected(signed_adj):
	signed_adj = signed_adj.tocsr()
	return sparse.csr_matrix.sign(signed_adj + signed_adj.T)

#Can be used for sec- implementation (run any unsigned method on pos_network and neg_network, and concatenate results)
def split_signed_network(signed_adj):
	signed_adj = signed_adj.tocoo()
	neg_edges = np.where(signed_adj.data == -1)[0]
	pos_edges = np.where(signed_adj.data == 1)[0]
	neg_network = sparse.csr_matrix((signed_adj.data[neg_edges], (signed_adj.row[neg_edges], signed_adj.col[neg_edges])), shape=signed_adj.shape)
	pos_network = sparse.csr_matrix((signed_adj.data[pos_edges], (signed_adj.row[pos_edges], signed_adj.col[pos_edges])), shape=signed_adj.shape)
	return pos_network, neg_network


#========UTILS FOR READING IN EMBEDDINGS FOR BASELINES IN COMMON FORMATS===========
def read_in_node2vec_format(emb_file, delimiter):
	print('----')
	representations_dict = {}
	representation_unorder = np.genfromtxt(emb_file, dtype=float, delimiter=delimiter, skip_header=1)
	m, n = representation_unorder.shape
	print('representation_unorder read in.')
	for i in range(m):
		if i % 50000 == 0:
			print(i)
		key = int(representation_unorder[i, 0])
		value = representation_unorder[i, 1:]
		representations_dict[key] = value
	return representations_dict

#Convert dict of embs to matrix
def emb2mat(representations_dict, N = None):
	if N is None:
		N = max(representations_dict.keys()) + 1 #max node ID (assume node IDs start from 0)
	D = len(representations_dict[ list(representations_dict.keys())[0] ])
	emb = np.zeros((N,D))
	for node_id in representations_dict.keys():
		emb[node_id,:] = representations_dict[node_id]
	return emb

#=========UTILS FOR READING SLASHDOT DATA===============
def read_slashdot_data():
	#List of trolls
	viz_list = np.loadtxt("data/slashdot-zoo.trolls").astype(int) #List of troll users https://github.com/gnemeuyil/DSG/blob/master/Slashdot/slashdot-zoo/out.trolls
	print("There are %d trolls:" % len(viz_list), viz_list)
	#Trolls
	nx_graph = nx.read_edgelist("data/slashdot-zoo.edgelist", comments="%", create_using = nx.DiGraph(), data=(('weight',float),)) #signed directed network

	return nx_graph, viz_list
