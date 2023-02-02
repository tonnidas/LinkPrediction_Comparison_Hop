# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle
import stellargraph as sg
import os
from stellargraph import StellarGraph, datasets

import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
from IPython.display import display, HTML

from hopInfo import addHopFeatures, addHopAdjacency

def prepare_hopped_graph(data_name, hop):
    # Get the original data from stellargraph
    if data_name == 'Cora':
        dataset = datasets.Cora()
        display(HTML(dataset.description))
        graph, _ = dataset.load(largest_connected_component_only=True, str_node_ids=True)
    if data_name == 'CiteSeer':
        dataset = datasets.CiteSeer()
        display(HTML(dataset.description))
        graph, _ = dataset.load(largest_connected_component_only=True)
    if data_name == 'PubMed':
        dataset = datasets.PubMedDiabetes()
        display(HTML(dataset.description))
        graph, _ = dataset.load()

    print(dataset)
    print("graph info for from stellargraph = ", graph.info())

    # Convert the graph into features and adj
    features = graph.node_features(nodes=None)
    features = sparse.csr_matrix(features)
    adj = graph.to_adjacency_matrix(nodes=None)
    adj = sparse.csr_matrix(adj)

    if hop != 0:
        # Manually store hopped info in pickle
        features = addHopFeatures(features, adj, hop)
        adj = addHopAdjacency(adj, hop + 1)

    print(type(adj))
    print(type(features))

    f1 = 'pickles/from_stellargraph/' + data_name + '_features_hop_' + str(hop) + '_stellergraph' + '.pickle'
    a1 = 'pickles/from_stellargraph/' + data_name + '_adj_hop_' + str(hop) + '_stellergraph' + '.pickle'

    with open(f1, 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(a1, 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Done storing hopped features and adj in sparse form")

# =======================================================================
data_name = 'CiteSeer'
hop_count = [3, 4]
for each_hop in hop_count: 
    prepare_hopped_graph(data_name, each_hop)
