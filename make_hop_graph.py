import networkx as nx
import numpy as np
import scipy.sparse 
import pickle5 as pickle
import stellargraph as sg
from stellargraph import StellarGraph, datasets

def get_sg_graph(adjPickleFile, featuresPickleFile):

    # adjPickleFile = 'pickles/{}_adj_hop_{}_stellergraph.pickle'.format(dataName, hop) # 'pickles/cora_adj_hop_3.pickle'
    # featuresPickleFile = 'pickles/{}_features_hop_{}_stellergraph.pickle'.format(dataName, hop) # 'pickles/cora_features_hop_3.pickle

    print('adjPickleFile:', adjPickleFile)
    print('featuresPickleFile', featuresPickleFile)

    # loading adj and features from pickle
    with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 
    with open(featuresPickleFile, 'rb') as handle: features = pickle.load(handle)

    print('adj shape:', adj.shape)
    print('feature shape:', features.shape)

    # make nx graph from scipy matrix
    nxGraph = nx.from_scipy_sparse_array(adj, parallel_edges=False, create_using=None, edge_attribute='weight')

    # add features to nx graph
    for node_id, node_data in nxGraph.nodes(data=True):
        node_feature = features[node_id].todense()
        node_data["feature"] = np.squeeze(np.asarray(node_feature)) # convert to 1D matrix to array

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="paper", edge_type_default="cites", node_features="feature")
    print(sgGraph.info())

    return sgGraph
