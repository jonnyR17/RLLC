import numpy as np
import time
import os
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
from torch_geometric.data import Batch
import torch_geometric as pyg
import tqdm
from collections import namedtuple
from itertools import product
from copy import deepcopy
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score
from neighbor_selector import NeighborSelector
from depth_selector import DepthSelector
from node_selector import NodeSelector
import pickle
class Newgraph(object):
    def __init__(self):
        self.khop=3

    def fake_graph(self,node_list,edge_list):
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_list,self.khop,edge_list)
        fake_graph = pyg.data.Data(x=subset,edge_index=edge_index)
        return fake_graph