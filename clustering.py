import networkx as nx
import numpy as np





class LouvainClustering:




    def __init__(self, correlation_matrix, resolution=1):
        self.resolution = resolution
        self.graph = nx.from_numpy_array(np.abs(correlation_matrix))
   

    def cluster(self, ):
        return nx.algorithms.community.louvain_communities(self.graph, resolution=i)