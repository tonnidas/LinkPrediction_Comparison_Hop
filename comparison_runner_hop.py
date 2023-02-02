# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import pickle
import os

from comparison import do
from make_hop_graph import get_sg_graph

data_name = 'cora'
split = 42
hop = 1

# outputFileName = "cora_original_0_stellergraph.txt"
# adjFileName = "pickles/cora_adj_original_0_stellergraph.pickle"
# featureFileName = "pickles/cora_features_original_0_stellergraph.pickle"

outputFileName = "cora_42_hop_1.txt"
adjFileName = "pickles/from_ARGA/cora_adj_hop_1.pickle"
featureFileName = "pickles/from_ARGA/cora_features_hop_1.pickle"

sgGraph = get_sg_graph(adjFileName, featureFileName)
df = do('custom', 42, sgGraph)

f1 = open(outputFileName, "w")
f1.write("For data_name: {}, split: {}, hop: {} --------------------\n".format(data_name, split, hop))
f1.write(df.to_string())
f1.close()
