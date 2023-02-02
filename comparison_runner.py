# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import scipy.sparse 
from scipy.sparse import csr_matrix
import pickle
import stellargraph as sg
import os
from stellargraph import StellarGraph, datasets

import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA

from compare_4_models import run


# Make the graph from the features and adj
def get_sg_graph(adj, features):

    print('adj shape:', adj.shape)
    print('feature shape:', features.shape)

    # make nx graph from scipy matrix
    nxGraph = nx.from_scipy_sparse_array(adj)

    # add features to nx graph
    for node_id, node_data in nxGraph.nodes(data=True):
        node_feature = features[node_id].todense()
        node_data["feature"] = np.squeeze(np.asarray(node_feature)) # convert to 1D matrix to array

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="paper", edge_type_default="cites", node_features="feature")
    print(sgGraph.info())

    return sgGraph

# =================================================================================================================================================

# Change and get result for expected set up

data_name = 'Cora' 
hop = 3
split_num = [42, 56, 61, 69, 73]  # split_num = [42, 56, 61, 69, 73]

# read adj and features from pickle and prepare sg graph

featurePickleFile = 'pickles/from_stellargraph/{}_features_hop_{}_stellergraph.pickle'.format(data_name, hop)
adjPickleFile = 'pickles/from_stellargraph/{}_adj_hop_{}_stellergraph.pickle'.format(data_name, hop)
with open(featurePickleFile, 'rb') as handle: features = pickle.load(handle) 
with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 

sgGraph = get_sg_graph(adj, features)

# =================================================================================================================================================

# run comparison for sg graph

for rand_state in split_num:

    outputDf = run(data_name, sgGraph, rand_state)

    outputFileName = "Result/from_stellargraph/All_Four/{}_{}_hop_{}.txt".format(data_name, rand_state, hop)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, rand_state, hop))
    f1.write(outputDf.to_string())
    f1.close()
    print("Done calculating node2vec, attri2vec, graphsage, gcn results for " + data_name + " with random state " + str(rand_state) + ", hop " + str(hop))