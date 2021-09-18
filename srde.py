import numpy as np

def learn_embeddings(adjmat, dim = 128, c = 0.98):
	prox_mat = get_prox(adjmat, c = c)
	emb = get_dist(prox_mat, dim = dim)
	return emb

#Get proximity scores of all nodes to other nodes using (exact) signed RWR
def get_prox(adjmat, c = 0.98):
	I = np.eye(adjmat.shape[0])
	prox_mat = (1-c)*np.linalg.inv(I - c*adjmat)
	return prox_mat

#Embed distribution of proximity scores for each node by creating row-wise histograms
def get_dist(mat, dim = 128):
	emb = np.apply_along_axis(lambda a: np.histogram(a, bins = dim)[0], 1, mat) #Bin each row
	return emb

if __name__ == "__main__":
	signed_adj = np.asarray([[0,1,-1,0], [1,0,-1,0], [-1,-1,0,1], [0,0,1,0]])
	emb = learn_embeddings(signed_adj, dim = 2)


