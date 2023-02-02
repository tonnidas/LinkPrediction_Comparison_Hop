# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import pickle
import os

from comparison import do

data_name = 'PubMedDiabetes'
split = 42

df = do(data_name, split)

outputFileName = "{}_{}.txt".format(data_name, split)
f1 = open(outputFileName, "w")
f1.write("For data_name: {}, split: {} --------------------\n".format(data_name, split))
f1.write(df.to_string())
f1.close()
