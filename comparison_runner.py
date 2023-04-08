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

from math import isclose
from sklearn.decomposition import PCA

from compare_4_models import run_4_models


# ================================================================================================================================================================
# Make the graph from the features and adj
def get_sg_graph(adj, features):
    print('adj shape:', adj.shape, 'feature shape:', features.shape)
    nxGraph = nx.from_scipy_sparse_array(adj)                           # make nx graph from scipy matrix

    # add features to nx graph
    for node_id, node_data in nxGraph.nodes(data=True):
        node_feature = features[node_id].todense()
        node_data["feature"] = np.squeeze(np.asarray(node_feature)) # convert to 1D matrix to array

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="paper", edge_type_default="cites", node_features="feature")
    print(sgGraph.info())

    return sgGraph
# ================================================================================================================================================================




# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--folder_name')
parser.add_argument('--dataset')
parser.add_argument('--hop')

# python comparison_runner.py --folder_name=Artificial_Rand_Graph --dataset=ArtificialV4 --hop=0   
args = parser.parse_args()
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Set up
folder_name = args.folder_name # 'Cora_CiteSeer_PubMed' or 'Artificial_Rand_Graph'
data_name = args.dataset       # 'Cora' or 'CiteSeer' or 'PubMed' or ''ArtificialV4
hop = int(args.hop)            # 1
split_num = [42]               # split_num = [42, 56, 61, 69, 73]
print('node2vec, attri2vec, graphsage, gcn results for ' + data_name + ', hop ' + str(hop))


# ================================================================================================================================================================
# read adj and features from pickle and prepare sg graph
featurePickleFile = '../graph-data/{}/Processed/{}_features_hop_{}.pickle'.format(folder_name, data_name, hop)
adjPickleFile = '../graph-data/{}/Processed/{}_adj_hop_{}.pickle'.format(folder_name, data_name, hop)
with open(featurePickleFile, 'rb') as handle: features = pickle.load(handle) 
with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 

sgGraph = get_sg_graph(adj, features)        # make the graph

# run Node2Vec, Attri2Vec, GraphSage, GCN models for sg graph
for rand_state in split_num:

    outputDf = run_4_models(data_name, sgGraph, rand_state)

    outputFileName = "results/{}_{}_hop_{}.txt".format(data_name, rand_state, hop)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, rand_state, hop))
    f1.write(outputDf.to_string())
    f1.close()
    print("Running comparison_runner: Done calculating node2vec, attri2vec, graphsage, gcn results for " + data_name + " with random state " + str(rand_state) + ", hop " + str(hop))
# ================================================================================================================================================================
