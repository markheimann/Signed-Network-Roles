"""
Feature sets we create:

1) Averaged structural properties
2) Flattened adjacency matrix
3) Adjacency matrix + regular graphlet counts
4) Adjacency matrix + signed graphlet counts
5) Signed graphlet kernels
"""
#import community
import numpy as np
from scipy import sparse
import networkx as nx
from utils import to_nx

from balance import *

def threshold_adjmat(adj_mat, theta):
    """
    Parameters
    ----------
    adj_mat : np.ndarray
        Adjacency matrix
    theta : float
        Absolute value below which adj_mat entries are set to 0
    Output
    ------
    Thresholded copy of adj_mat
    """
    adj_mat_copy = np.copy(adj_mat)
    adj_mat_copy[np.abs(adj_mat_copy) < theta] = 0
    return adj_mat_copy

## 1) Averaged structural properties

def avg_weighted_degree(G):
    """
    Parameters
    ----------
    G : networkx Graph
        Input graph
    Output
    ------
    Average weighted degree of graph
    """
    if not G.number_of_nodes():
        return 0
    degree_sum = sum([pair[1] for pair in G.degree(weight='weight')])
    return degree_sum / G.number_of_nodes()

def structural_features(G):
    """
    Parameters
    ----------
    G : networkx Graph
        Input graph
    Output
    ------
    Structural feature vector of 
    [
    """
    features = [
        nx.density(G),
        avg_weighted_degree(G),
        nx.average_clustering(G),
        nx.average_shortest_path_length(G),
        # modularity(G)
    ]
    
    return np.array(features)

## 2) Flattened adjacency matrix

def flatten_triu(X):
    """
    Parameters
    ----------
    X : np.ndarray
        Data matrix
    Output
    ------
    Upper triangular of X, not including the diagonal, flattened
    into a one-dimensional array. Turns a symmetric matrix into a feature vector
    """
    return np.ravel(X[np.triu_indices_from(X, k=1)])

## 3) Adjacency matrix + regular graphlet counts

def all_triads(G, complete = True):
    """
    Parameters
    ----------
    G : networkx Graph
        Input graph
    complete : boolean
        Specifies whether to only collect complete triads or not
    Output
    ------
    List of triplets, which are all triangles in the input graph
    """
    nodes = list(G.nodes())
    
    triads = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                a, b, c = nodes[i], nodes[j], nodes[k]
                if complete:
                    if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a):
                        triads.append((a, b, c))
                else:
                    triads.append((a, b, c))                    
    return triads

### TODO

## 4) Adjacency matrix + signed graphlet counts

def edge_sign(G, a, b):
    """
    Parameters
    ----------
    G : networkx Graph
        Input graph
    a : int
        Node ID in G
    b : int
        Node ID in G
    Output
    ------
    Returns the sign of the edge between u and v:
    -1 if negative, 0 if nonexistent, 1 if positive
    """
    if not G.has_edge(a, b):
        return 0
    return 1 if G[a][b]['weight'] > 0 else -1

def count_configs(G, triads):
    if len(G) == 0: raise ValueError("empty graph")
    """
    Parameters
    ----------
    G : networkx Graph
        Input graph
    triads : list
        List of all triangles in G
    Output
    ------
    Vector of normalized counts of different triangle types in G
    """
    config_counts = np.zeros(len(configs))
    if len(triads) == 0: return config_counts #e.g. graph smaller than 3 nodes
    for (a, b, c) in triads:
        # need to account for nodes not there
        config = tuple(sorted([
            edge_sign(G, a, b),
            edge_sign(G, b, c),
            edge_sign(G, c, a)
        ]))
        if config in config_dict:
            config_counts[config_dict[config]] += 1
    return config_counts / np.sum(config_counts)

## 5) Signed graphlet kernels
def signed_graphlet_kernel(G):
    """
    Parameters
    ----------
    G : networkx Graph
        Input graph
    triads : list
        List of all triangles in G
    Output
    ------
    Vector of normalized counts of different triangle types in G
    """
    triads = all_triads(G, complete = False)
    features = count_configs(G, triads)
    return features

